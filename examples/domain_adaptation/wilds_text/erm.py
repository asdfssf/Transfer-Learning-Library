import argparse
import os
import shutil
import time
import pprint
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy
from transformers import AdamW, get_linear_schedule_with_warmup

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

import wilds
from wilds.common.grouper import CombinatorialGrouper

import utils
from utils import DistilBertClassifier
from tllib.utils.logger import CompleteLogger
from tllib.utils.meter import AverageMeter
from tllib.utils.metric import accuracy


def main(args):
    logger = CompleteLogger(args.log, args.phase)
    writer = SummaryWriter(args.log)
    pprint.pprint(args)

    if args.local_rank == 0:
        print("opt_level = {}".format(args.opt_level))
        print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
        print("loss_scale = {}".format(args.loss_scale, type(args.loss_scale)))

        print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    cudnn.benchmark = True
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.set_printoptions(precision=10)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='ncc1',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    # Data loading code
    train_transform = utils.get_transform(args.arch, args.max_token_length)
    val_transform = utils.get_transform(args.arch, args.max_token_length)

    train_labeled_dataset, train_unlabeled_dataset, test_datasets, args.num_classes, args.class_names, labeled_dataset = \
        utils.get_dataset(args.data, args.data_dir, args.unlabeled_list, args.test_list,
                          train_transform, val_transform, verbose=args.local_rank == 0)

    # create model
    if args.local_rank == 0:
        print("=> using pre-trained model '{}'".format(args.arch))
    model = DistilBertClassifier.from_pretrained(args.arch, num_labels=args.num_classes)
    model = model.cuda().to()

    # Data loading code
    train_labeled_sampler = None
    train_unlabeled_sampler = None
    if args.distributed:
        train_labeled_sampler = DistributedSampler(train_labeled_dataset)
        train_unlabeled_sampler = DistributedSampler(train_unlabeled_dataset)
    elif args.uniform_over_groups:
        train_grouper = CombinatorialGrouper(dataset=labeled_dataset, groupby_fields=args.groupby_fields)
        groups, group_counts = train_grouper.metadata_to_group(train_labeled_dataset.metadata_array, return_counts=True)
        group_weights = 1 / group_counts
        weights = group_weights[groups]
        train_labeled_sampler = WeightedRandomSampler(weights, len(train_labeled_dataset), replacement=True)

    train_labeled_loader = DataLoader(
        train_labeled_dataset, batch_size=args.batch_size[0], shuffle=(train_labeled_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_labeled_sampler
    )
    train_unlabeled_loader = DataLoader(
        train_unlabeled_dataset, batch_size=args.batch_size[1], shuffle=(train_labeled_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_unlabeled_sampler
    )

    no_decay = ['bias', 'LayerNorm.weight']
    params = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(params, lr=args.lr)
    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    # model, optimizer = amp.initialize(model, optimizer,
    #                                   opt_level=args.opt_level,
    #                                   keep_batchnorm_fp32=args.keep_batchnorm_fp32,
    #                                   loss_scale=args.loss_scale
    #                                   )
    lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                   num_training_steps=len(train_labeled_loader) * args.epochs,
                                                   num_warmup_steps=0)
    lr_scheduler.step_every_batch = True
    lr_scheduler.use_metric = False

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda()

    if args.phase == 'test':
        # resume from the latest checkpoint
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        model.load_state_dict(checkpoint)
        for n, d in zip(args.test_list, test_datasets):
            if args.local_rank == 0:
                print(n)
            utils.validate(d, model, -1, writer, args)
        return

    best_val_metric = 0
    best_test_metric = 0
    for epoch in range(args.epochs):

        lr_scheduler.step(epoch)
        if args.local_rank == 0:
            print(lr_scheduler.get_last_lr())
            writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[-1], epoch)
        # train for one epoch
        train(train_labeled_loader, model, criterion, optimizer, epoch, writer, args)
        # evaluate on validation set
        for n, d in zip(args.test_list, test_datasets):
            if args.local_rank == 0:
                print(n)
            if n == 'val':
                val_metric = utils.validate(d, model, epoch, writer, args)
            elif n == 'test':
                test_metric = utils.validate(d, model, epoch, writer, args)

        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            is_best = val_metric > best_val_metric
            best_val_metric = max(val_metric, best_val_metric)
            torch.save(model.state_dict(), logger.get_checkpoint_path('latest'))
            if is_best:
                best_test_metric = test_metric
                shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
    print("best val performance: {:.3f}\n test performance: {:.3f}".format(best_val_metric, best_test_metric))


def train(train_loader, model, criterion, optimizer, epoch, writer, args):
    batch_time = AverageMeter('Time', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    top1 = AverageMeter('Top 1', ':3.1f')

    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, target, metadata) in enumerate(train_loader):

        # compute output
        output = model(input.cuda())
        loss = criterion(output, target.cuda())

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
            # scaled_loss.backward()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            prec1, = accuracy(output.data, target.cuda(), topk=(1,))

            # Average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                prec1 = utils.reduce_tensor(prec1, args.world_size)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            global_step = epoch * len(train_loader) + i

            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            if args.local_rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader),
                    args.world_size * args.batch_size[0] / batch_time.val,
                    args.world_size * args.batch_size[0] / batch_time.avg,
                    batch_time=batch_time,
                    loss=losses, top1=top1))


if __name__ == '__main__':
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith('__')
                         and callable(models.__dict__[name]))
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Dataset parameters
    parser.add_argument('data_dir', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='civilcomments', choices=wilds.supported_datasets,
                        help='dataset: ' + ' | '.join(wilds.supported_datasets) +
                             ' (default: civilcomments)')
    parser.add_argument('--unlabeled-list', nargs='+', default=["test_unlabeled", ])
    parser.add_argument('--test-list', nargs='+', default=["val", "test"])
    parser.add_argument('--metric', default='acc_wg')
    # model parameters
    parser.add_argument('--arch', '-a', metavar='ARCH', default='distilbert-base-uncased',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet50)')
    parser.add_argument('--max_token_length', type=int, default=300)
    # Learning rate schedule parameters
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR',
                        help='Initial learning rate.  Will be scaled by <global batch size>/256: '
                             'args.lr = args.lr*float(args.batch_size*args.world_size)/256')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1r-5)')
    # training parameters
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=5, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=(16, 16), type=int, nargs='+',
                        metavar='N', help='mini-batch size per process for source'
                                          ' and target domain (default: (64, 64))')
    parser.add_argument('--print-freq', '-p', default=200, type=int,
                        metavar='N', help='print frequency (default: 200)')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')
    parser.add_argument('--opt-level', type=str)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--log', type=str, default='src_only',
                        help='Where to save logs, checkpoints and debugging images.')
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis'm only analysis the model.")
    parser.add_argument('--uniform_over_groups', action='store_true',
                        help='sample examples such that batches are uniform over groups')
    parser.add_argument('--groupby_fields', nargs='+')
    args = parser.parse_args()
    main(args)
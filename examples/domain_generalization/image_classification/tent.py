"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import time
import random
import warnings
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader

sys.path.append('../../..')
from dglib.generalization.tent import Tent, configure_model, collect_params
from dglib.modules.classifier import ImageClassifier as Classifier
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger

sys.path.append('.')
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    val_transform = utils.get_val_transform(args.val_resizing)
    print("transform: ", val_transform)

    test_dataset, num_classes = utils.get_dataset(dataset_name=args.data, root=args.root, task_list=args.targets,
                                                  split='test', download=True, transform=val_transform, seed=args.seed)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    print("test_dataset_size: ", len(test_dataset))

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = Classifier(backbone, num_classes, pool_layer=pool_layer).to(device)

    # load pretrained parameters
    checkpoint = torch.load(args.pretrained)
    classifier.load_state_dict(checkpoint)
    acc1_origin = utils.validate(test_loader, classifier, args, device)
    print("test acc before adaptation = {}".format(acc1_origin))

    # adjust bn layers and collect parameters to update
    classifier = configure_model(classifier)
    params, param_names = collect_params(classifier)

    if args.phase == 'test':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)
        acc1 = utils.validate(test_loader, classifier, args, device)
        print("test acc after adaptation = {}".format(acc1))
        return

    optimizer = Adam(params, args.lr, weight_decay=args.wd)
    tent_classifier = Tent(classifier, optimizer, args.optim_steps)

    print(f"classifier for adaptation: %s", classifier)
    print(f"params for adaptation: %s", param_names)
    print(f"optimizer for adaptation: %s", optimizer)

    acc1 = test_time_adaptation(test_loader, tent_classifier, args)
    print("test acc after adaptation = {}".format(acc1))

    # save results
    torch.save(classifier.state_dict(), logger.get_checkpoint_path('best'))
    logger.close()


def test_time_adaptation(val_loader, model, args) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # notice model is in train-mode instead of eval-mode because we need to calculate batch statistics according to
    # current batch
    with torch.no_grad():
        end = time.time()
        for i, (images, target, _) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target)[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} '.format(top1=top1))

    return top1.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tent for Domain Generalization')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='PACS',
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: PACS)')
    parser.add_argument('-t', '--targets', nargs='+', default=None,
                        help='target domain(s)')
    parser.add_argument('--val-resizing', type=str, default='default')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet50)')
    parser.add_argument('--no-pool', action='store_true', help='no pool layer after the feature extractor.')
    # training parameters
    parser.add_argument('--pretrained', type=str, help='pretrained model')
    parser.add_argument('--optim-steps', type=int, default=1, help='steps to update parameters for each batch')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing testing. ')
    parser.add_argument("--log", type=str, default='tent',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model.")
    args = parser.parse_args()
    main(args)

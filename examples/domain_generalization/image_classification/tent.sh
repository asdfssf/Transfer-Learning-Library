#!/usr/bin/env bash
# ResNet50, PACS
CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s A C S -t P -a resnet50 --freeze-bn --seed 0 --log logs/baseline/PACS_P
CUDA_VISIBLE_DEVICES=0 python tent.py data/PACS -d PACS -t P -a resnet50 --seed 0 --log logs/tent/PACS_P \
  --pretrained logs/baseline/PACS_P/checkpoints/best.pth

CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P C S -t A -a resnet50 --freeze-bn --seed 0 --log logs/baseline/PACS_A
CUDA_VISIBLE_DEVICES=0 python tent.py data/PACS -d PACS -t A -a resnet50 --seed 0 --log logs/tent/PACS_A \
  --pretrained logs/baseline/PACS_A/checkpoints/best.pth

CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A S -t C -a resnet50 --freeze-bn --seed 0 --log logs/baseline/PACS_C
CUDA_VISIBLE_DEVICES=0 python tent.py data/PACS -d PACS -t C -a resnet50 --seed 0 --log logs/tent/PACS_C \
  --pretrained logs/baseline/PACS_C/checkpoints/best.pth

CUDA_VISIBLE_DEVICES=0 python baseline.py data/PACS -d PACS -s P A C -t S -a resnet50 --freeze-bn --seed 0 --log logs/baseline/PACS_S
CUDA_VISIBLE_DEVICES=0 python tent.py data/PACS -d PACS -t S -a resnet50 --seed 0 --log logs/tent/PACS_S \
  --pretrained logs/baseline/PACS_S/checkpoints/best.pth

# ResNet50, Office-Home
CUDA_VISIBLE_DEVICES=0 python baseline.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr -a resnet50 --seed 0 --log logs/baseline/OfficeHome_Pr
CUDA_VISIBLE_DEVICES=0 python tent.py data/office-home -d OfficeHome -t Pr -a resnet50 --seed 0 --log logs/tent/OfficeHome_Pr \
  --pretrained logs/baseline/OfficeHome_Pr/checkpoints/best.pth

CUDA_VISIBLE_DEVICES=0 python baseline.py data/office-home -d OfficeHome -s Ar Cl Pr -t Rw -a resnet50 --seed 0 --log logs/baseline/OfficeHome_Rw
CUDA_VISIBLE_DEVICES=0 python tent.py data/office-home -d OfficeHome -t Rw -a resnet50 --seed 0 --log logs/tent/OfficeHome_Rw \
  --pretrained logs/baseline/OfficeHome_Rw/checkpoints/best.pth

CUDA_VISIBLE_DEVICES=0 python baseline.py data/office-home -d OfficeHome -s Ar Rw Pr -t Cl -a resnet50 --seed 0 --log logs/baseline/OfficeHome_Cl
CUDA_VISIBLE_DEVICES=0 python tent.py data/office-home -d OfficeHome -t Cl -a resnet50 --seed 0 --log logs/tent/OfficeHome_Cl \
  --pretrained logs/baseline/OfficeHome_Cl/checkpoints/best.pth

CUDA_VISIBLE_DEVICES=0 python baseline.py data/office-home -d OfficeHome -s Cl Rw Pr -t Ar -a resnet50 --seed 0 --log logs/baseline/OfficeHome_Ar
CUDA_VISIBLE_DEVICES=0 python tent.py data/office-home -d OfficeHome -t Ar -a resnet50 --seed 0 --log logs/tent/OfficeHome_Ar \
  --pretrained logs/baseline/OfficeHome_Ar/checkpoints/best.pth

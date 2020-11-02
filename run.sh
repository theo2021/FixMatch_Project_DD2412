#!/bin/bash
export SDIR = "/afs/pdc.kth.se/home/t/thepan/Public/"
python3 main.py --db_dir="$SDIR" --dataset='CIFAR10' --labels_per_class=25 --batch_size=64 --mu=7 --ct_augment_update=10 --workers=20 --steps=1000 --model_save_dir="$SDIR/models" --tb_logdir="$SDIR/tensorboard"
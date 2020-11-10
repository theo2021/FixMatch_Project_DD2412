
#!/bin/bash
export SDIR="/home/fredericonesti"
python main.py --db_dir="$SDIR" --dataset='CIFAR10'  --model_save_dir="$SDIR/EXPERIMENT_1" --tb_logdir="$SDIR/EXPERIMENT_1" \
                --labels_per_class=25 \
                --batch_size=64 \
                --mu=7 \
                --ct_augment_update=10 \
                --workers=25 \
                --steps=1048576 \
                --validation_percentage=0.1 \
                --seed=8721987 \
                --scheduler_steps=1048576 \
                --warmup_scheduler=0.01 \
                --lr=0.03 \
                --momentum=0.9 \
                --nesterov=True \
                --weight_decay=0.0005 \
                --threshold=0.95 \
                --ct_augment_batch_size=64 \
                --ema=True \
                --gcross_thresh=0.7 \
                --loss_func="cross,cross"

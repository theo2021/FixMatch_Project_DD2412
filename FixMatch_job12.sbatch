#!/usr/bin/env bash
#SBATCH --mem  20GB
#SBATCH --gres gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task 10
#SBATCH --time 48:00:00
#SBATCH --constrain="gondor|balrog|rivendell|smaug|belegost|shire|khazadum"
#SBATCH --mail-user afnl@kth.se
#SBATCH --output /Midgard/home/afnl/logs/%J_slurm.out
#SBATCH --error  /Midgard/home/afnl/logs/%J_slurm.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. /Midgard/home/afnl/miniconda3/etc/profile.d/conda.sh
conda activate smarc_env
rm -r /local_storage/datasets/afnl/job12
mkdir /local_storage/datasets/afnl/job12
mkdir /local_storage/datasets/afnl/fixmatch_data
python3 /Midgard/home/afnl/DD2412_finalproj/reorganization/main.py --db_dir='/local_storage/datasets/afnl/fixmatch_data' --dataset='CIFAR10' --labels_per_class=4 --batch_size=64 --mu=7 --ct_augment_update=15 --ct_augment_batch_size=64 --workers=20 --validation_percentage=0.1 --seed=1 --steps=100000 --scheduler_steps=100000 --warmup_scheduler=0.01 --lr=0.03 --momentum=0.9 --nesterov=True --weight_decay=0.0005 --threshold=0.95 --model_save_dir='/local_storage/datasets/afnl/job12/' --tb_logdir='/local_storage/datasets/afnl/job12/'
cp -r /local_storage/datasets/afnl/job12 /Midgard/home/afnl/Results




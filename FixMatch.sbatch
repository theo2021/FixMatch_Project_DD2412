#!/usr/bin/env bash
#SBATCH --mem  20GB
#SBATCH --gres gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task 20
#SBATCH --time 48:00:00
#SBATCH --constrain="balrog|gondor"
#SBATCH --mail-user afnl@kth.se
#SBATCH --output /Midgard/home/afnl/logs/%J_slurm.out
#SBATCH --error  /Midgard/home/afnl/logs/%J_slurm.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. /Midgard/home/afnl/miniconda3/etc/profile.d/conda.sh
conda activate smarc_env
rm -r /local_storage/datasets/afnl/fixmatch_save
mkdir /local_storage/datasets/afnl/fixmatch_save
mkdir /local_storage/datasets/afnl/fixmatch_data
python3 /Midgard/home/afnl/DD2412_finalproj/FIXMATCHPROJ_vs2/train_optimize/train.py --root='/local_storage/datasets/afnl/fixmatch_data' --download=True --use_database='CIFAR10' --task='train' --save_dir='/local_storage/datasets/afnl/fixmatch_save/' --B=64 --K=100000 --mu=7 --tbsw_logdir='/local_storage/datasets/afnl/fixmatch_save/' --labels_per_class=25 --confidence_threshold=15 --lr=0.03 --momentum=0.9 --weight_decay=0.0005 --warmup_scheduler=0.01 --run_ctaugment=15 --num_workers1=5 --num_workers2=5 --num_workers3=5 --cifar_what=10
cp -r /local_storage/datasets/afnl/fixmatch_save /Midgard/home/afnl/Results




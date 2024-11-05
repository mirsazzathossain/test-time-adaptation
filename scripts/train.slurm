#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=0-2
#SBATCH --ntasks=1 
#SBATCH -t 1-00:00:00 
#SBATCH -p gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --constraint=a40
#SBATCH -N 1 
#SBATCH --mem=256G
#SBATCH --output=log/slurm/tta/log-%A-%a.log 
#SBATCH -J tta


export TORCH_HOME=/project/CollabRoboGroup/.cache
export TRANSFORMERS_CACHE=/project/CollabRoboGroup/.cache
export PYTHONPATH=/opt/conda/lib/python3.10/site-packages:.

module purge
module load apptainer

domains=('continual' 'mixed_domains' 'gradual')
domain=${domains[${SLURM_ARRAY_TASK_ID}]}


echo "job name: Test-time-$domain-adaptation"

apptainer exec --nv /scratch/mi8uu/mrm/sifs/tta_latest.sif python test_time.py \
    --cfg cfgs/imagenet_c/ours.yaml SETTING $domain CORRUPTION.NUM_EX 50000

#!/bin/bash
#SBATCH --job-name=finetune-en-word-whole
#SBATCH --output=/swdata/yin/Cui/EM/reveil/slurm_logs/en-word-whole-%j.out  # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/swdata/yin/Cui/EM/reveil/slurm_logs/en-word-whole-%j.err  # where to store error messages
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Send some noteworthy information to the output log
SlurmID=$SLURM_JOBID
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# English
python /swdata/yin/Cui/EM/reveil/train_qwen2-5-vl-reveil-whole-en-word.py

                                        
# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0

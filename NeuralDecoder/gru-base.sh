#!/bin/bash

#SBATCH --job-name=example                    # Submit a job named "example"
#SBATCH --nodes=1                             # Using 1 node
#SBATCH --gres=gpu:1                          # Using 1 gpu
#SBATCH --time=0-04:00:00                     # 1 hour timelimit
#SBATCH --mem=10000MB                         # Using 10GB CPU Memory
#SBATCH --partition=class2                         # Using "b" partition 
#SBATCH --cpus-per-task=8                     # Using 4 maximum processor

source ${HOME}/.bashrc
source ${HOME}/anaconda3/bin/activate
conda activate pj

srun python3 -m neuralDecoder.main \
    dataset=speech_release_baseline \
    model=gru_stack_inputNet \
    learnRateDecaySteps=10000 \
    nBatchesToTrain=10000  \
    learnRateStart=0.02 \
    model.nUnits=1024 \
    model.stack_kwargs.kernel_size=32 \
    outputDir=/home/s2/nlp002/pj_data/derived/gru-base/baselineRelease
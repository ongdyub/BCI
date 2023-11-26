#!/bin/bash

#SBATCH --job-name=rnn-bi                    # Submit a job named "example"
#SBATCH --nodes=1                             # Using 1 node
#SBATCH --gres=gpu:2                          # Using 1 gpu
#SBATCH --time=0-05:00:00                     # 1 hour timelimit
#SBATCH --mem=50000MB                         # Using 10GB CPU Memory
#SBATCH --partition=class2                         # Using "b" partition 
#SBATCH --cpus-per-task=4                     # Using 4 maximum processor JOB ID : 122798

source ${HOME}/.bashrc
source ${HOME}/anaconda3/bin/activate
conda activate pj

srun python3 -m neuralDecoder.main \
    dataset=speech_release_baseline \
    model=gru_stack_inputNet \
    learnRateDecaySteps=10000 \
    nBatchesToTrain=10000  \
    learnRateStart=0.02 \
    model.stack_kwargs.kernel_size=32 \
    outputDir=/home/s2/nlp002/pj_data/derived/rnn-bi/baselineRelease
#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:v100-32:1
#SBATCH --output=/jet/home/dsouzare/out.txt
#SBATCH --error=/jet/home/dsouzare/err.txt

# activate conda
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate minitorch

nvidia-smi

cd $PROJECT
pwd
ls
cd llmsys_s24_hw2/llmsys_s24_hw2
pwd
ls
python -m pytest -l -v -m a2_1
python -m pytest -l -v -m a2_2
python -m pytest -l -v -m a2_3
python -m pytest -l -v -m a2_4
#python3 /jet/home/dsouzare/llmsys_s24_hw2/project/run_machine_translation.py
python3 project/run_machine_translation.py

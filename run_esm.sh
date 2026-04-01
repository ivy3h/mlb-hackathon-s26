#!/bin/bash
#SBATCH --job-name=esm_scores
#SBATCH --output=/coc/pskynet6/jhe478/logs/esm_scores_%j.log
#SBATCH --partition=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=starrysky,heistotron,deebot,nestor,cheetah,chitti,tachikoma,optimistprime,uniblab,puma,perseverance,clippy,xaea-12,megazord,trublu,omgwth,robby,protocol,crushinator,johnny5,qt-1,spd-13,alexa,cortana

source ~/.bashrc

cd /coc/pskynet6/jhe478/7850/project

echo "Python: $(which python3)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 compute_esm_scores.py

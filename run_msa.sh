#!/bin/bash
#SBATCH --job-name=msa_scores
#SBATCH --partition=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/msa_scores_%j.log
#SBATCH --exclude=starrysky,heistotron,deebot,nestor,cheetah,chitti,tachikoma,optimistprime,uniblab,puma,perseverance,clippy,xaea-12,megazord,trublu,omgwth,protocol,robby,crushinator,johnny5,qt-1,bishop,spd-13,hal

cd /coc/pskynet6/jhe478/7850/project
mkdir -p logs

echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

python compute_msa_scores.py

echo "End: $(date)"

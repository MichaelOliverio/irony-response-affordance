#!/bin/bash
#SBATCH -o Snakefile-ces-ukr.out
#SBATCH -e Snakefile-ces-ukr.out
#SBATCH --gres=gpu:1
#SBATCH -p gracehopper


# mettere SBATCH -p epito se si vuole usare epito oppure gracehopper se si vuole usare gracehopper

# mettere sbatch --reservation mike quando prenoto

source ~/mambaforge/etc/profile.d/conda.sh

# Attiva l'ambiente
# conda activate llm
conda activate llm_py312AIMLplus

# Esporta il PYTHONPATH di PyTorch
export HPCX_HOME=/opt/hpcx
export PYTHONPATH=/opt/pytorch/lib/python3.12/site-packages

# Verifica PyTorch e CUDA
python -c "import torch; print('Torch Version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Compiled with CUDA:', torch.version.cuda)"

# Lancia il tuo script
python 5_sample_dataset_automatic_eval.py

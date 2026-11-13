#!/bin/bash
#SBATCH --account=prasanna_933
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=a40:1
#SBATCH --time=00:10:00
#SBATCH --output=results/gpu_check_%j.log

nvidia-smi
source carc/env_setup.sh
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=16       # number of threads per task
#SBATCH --time 1:00:00          # format: HH:MM:SS

#SBATCH -A rorland1

module load autoload profile/deeplrn culturax/2309

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12340

#srun python mddp.py
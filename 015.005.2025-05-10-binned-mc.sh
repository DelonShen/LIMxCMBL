#!/bin/bash
acc="kipac:default"
partition="ampere"
time_limit="3:00:00"
mem_per_node="128G"

date=$(date +%Y-%m-%d)

for n_runs in $(seq 1560702 1560984); do
  job_name="015.005-${n_runs}"
    sbatch << EOF
#!/bin/bash

#SBATCH --job-name=${job_name}
#SBATCH --account=${acc}
#SBATCH --output="logs/${date}-${job_name}.out"
#SBATCH --error="logs/${date}-${job_name}.err"
#SBATCH --time=${time_limit}
#SBATCH --partition='${partition}'
#SBATCH --nodes=1
#SBATCH --mem=${mem_per_node}
#SBATCH --gpus=1

python 015.005.2025-05-10-binned-mc.py ${n_runs}

EOF
    echo ${job_name}
done

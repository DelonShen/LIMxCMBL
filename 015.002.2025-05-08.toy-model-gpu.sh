#!/bin/bash
acc="kipac:kipac"
partition="ada"
time_limit="8:00:00"


mem_per_node="128G"

date=$(date +%Y-%m-%d)



#80 iteration/s
#24hr -> 86400 secs
#-> 6912000 in 24hr
for n_runs in $(seq 6912000 6912009); do
#for n_runs in $(seq 720000 720001); do
#for n_runs in $(seq 5 5); do
  job_name="015.002-${n_runs}"
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

python 015.002.2025-05-08.toy-model-gpu.py ${n_runs}

EOF
    echo ${job_name}
done

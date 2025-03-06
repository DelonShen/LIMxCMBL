#!/bin/bash
lambda_values=$(python3 -c '
import numpy as np
lambda_idxs = np.arange(25)[15:-1][::-1]
print("\n".join(map(str, lambda_idxs)))
')

readarray -t lambda_idxs <<< "$lambda_values"

nb=100
nmc=250

date=$(date +%Y-%m-%d)
output_dir="logs"

# Set the Slurm parameters
partition="owners"
time_limit="24:00:00"
num_nodes=1
mem_per_node="64G"
cpus_per_task=1
output_dir="logs"

mkdir -p ${output_dir}

date=$(date +%Y-%m-%d)

for lambda_idx in "${lambda_idxs[@]}"; do
  for ell_idx in $(seq 0 99); do
    echo $lambda_idx
    lambda_formatted=$(echo $lambda_idx | tr '.' 'p')
    
    job_name="010.016-${lambda_formatted}-nb-${nb}-l-${ell_idx}-nmc-${nmc}"
    output_file="${output_dir}/${date}-${job_name}.out"
    error_file="${output_dir}/${date}-${job_name}.err"

    sbatch << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${output_file}
#SBATCH --error=${error_file}
#SBATCH --time=${time_limit}
#SBATCH -p ${partition}
#SBATCH --nodes=${num_nodes}
#SBATCH --mem=${mem_per_node}
#SBATCH --cpus-per-task=${cpus_per_task}
#SBATCH -G 1
#SBATCH -C GPU_MEM:48GB


python -u 010.016.2025-03-05-comb-mc-sample-bin.py ${lambda_idx} ${nb} ${ell_idx} ${nmc}

EOF
    echo ${job_name}
  done
done

echo "All jobs submitted"

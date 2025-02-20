#!/bin/bash
log2=13
lambda_values=$(python3 -c '
import numpy as np
lambda_idxs = np.arange(25)
print("\n".join(map(str, lambda_idxs)))
')

readarray -t lambda_idxs <<< "$lambda_values"

# Set the Slurm parameters
partition="kipac"
time_limit="24:00:00"
num_nodes=1
mem_per_node="64G"
cpus_per_task=1
output_dir="logs"

mkdir -p ${output_dir}

date=$(date +%Y-%m-%d)

for lambda_idx in "${lambda_idxs[@]}"; do
    echo $lambda_idx
    lambda_formatted=$(echo $lambda_idx | tr '.' 'p')
    
    job_name="dpss-diag-idx-${lambda_formatted}-log2-${log2}"
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

python -u 005.013.2025-02-19-dpss-diag-approx.py ${lambda_idx} ${log2}
EOF
    echo ${job_name}
done

echo "All jobs submitted"

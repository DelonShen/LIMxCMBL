#!/bin/bash
lambda_values=$(python3 -c '
import numpy as np
lambda_idxs = np.arange(25)
print("\n".join(map(str, lambda_idxs)))
')

log2=14
nbins=100
readarray -t lambda_idxs <<< "$lambda_values"

# Set the Slurm parameters
partition="kipac"
time_limit="24:00:00"
num_nodes=1
mem_per_node="192G"
cpus_per_task=1
output_dir="logs"

mkdir -p ${output_dir}

date=$(date +%Y-%m-%d)

for lambda_idx in "${lambda_idxs[@]}"; do
    echo $lambda_idx
    lambda_formatted=$(echo $lambda_idx | tr '.' 'p')
    
    job_name="bin-cov-idx-${lambda_formatted}-log2-${log2}-nbins-${nbins}"
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

python -u 009.001.2025-02-19-bin-cov.py ${lambda_idx} ${log2} ${nbins}

EOF
    echo ${job_name}
done

echo "All jobs submitted"

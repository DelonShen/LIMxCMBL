#!/bin/bash
lambda_values=$(python3 -c '
import numpy as np
lambdas = np.logspace(-5, 0, 50)
lambda_idxs = np.where((lambdas > 8e-4) & (lambdas < 2e-1))[0]
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
    
    job_name="snr-per-mode-mpmath-Lambda-idx-${lambda_formatted}-factor-4"
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

python -u 005.010.2025-02-18-SNR-diagonal-noise-dominated-CCAT-prime-mpmath-noise.py ${lambda_idx}
EOF
    echo "Submitted job for Lambda idx = ${lambda_idx}"
done

echo "All jobs submitted"

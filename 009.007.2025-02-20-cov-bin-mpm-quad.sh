#!/bin/bash
lambda_values=$(python3 -c '
import numpy as np
lambda_idxs = np.arange(31)
print("\n".join(map(str, lambda_idxs)))
')

nbins=100
readarray -t lambda_idxs <<< "$lambda_values"

date=$(date +%Y-%m-%d)
output_dir="logs"


#for lambda_idx in "${lambda_idxs[@]}"; do
#    echo $lambda_idx
#    lambda_formatted=$(echo $lambda_idx | tr '.' 'p')
#    job_name="bin-cov-idx-${lambda_formatted}-quad-nbins-${nbins}"
#    output_file="${output_dir}/${date}-${job_name}.out"
#    echo $output_file
#
#    python -u 009.004.2024-02-20-cov-bin-quad.py ${lambda_idx} ${nbins} &> ${output_file}
#
#done

# Set the Slurm parameters
partition="kipac"
time_limit="24:00:00"
num_nodes=1
mem_per_node="64G"
cpus_per_task=32
output_dir="logs"

mkdir -p ${output_dir}

date=$(date +%Y-%m-%d)

for lambda_idx in "${lambda_idxs[@]}"; do
    echo $lambda_idx
    lambda_formatted=$(echo $lambda_idx | tr '.' 'p')
    
    job_name="mpmath-bin-cov-idx-${lambda_formatted}-quad-nbins-${nbins}"
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

python -u 009.007.2024-02-20-cov-bin-mpm-quad.py ${lambda_idx} ${nbins}

EOF
    echo ${job_name}
done

echo "All jobs submitted"

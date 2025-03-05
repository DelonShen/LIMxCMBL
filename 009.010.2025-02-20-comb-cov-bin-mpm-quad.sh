#!/bin/bash
lambda_values=$(python3 -c '
import numpy as np
lambda_idxs = np.arange(25)[::-1]
lambda_idxs = [7]
print("\n".join(map(str, lambda_idxs)))
')

nbins=100
readarray -t lambda_idxs <<< "$lambda_values"

date=$(date +%Y-%m-%d)
output_dir="logs"



# Set the Slurm parameters
partition="kipac"
time_limit="72:00:00"
num_nodes=1
mem_per_node="64G"
cpus_per_task=32
output_dir="logs"

mkdir -p ${output_dir}

date=$(date +%Y-%m-%d)

for lambda_idx in "${lambda_idxs[@]}"; do
    echo $lambda_idx
    lambda_formatted=$(echo $lambda_idx | tr '.' 'p')
    
    job_name="comb-mpmath-quad-cov-idx-${lambda_formatted}-nbins-${nbins}"
    output_file="${output_dir}/${date}-${job_name}.out"
    error_file="${output_dir}/${date}-${job_name}.err"

#######################
#    output_file="${output_dir}/${date}-${job_name}"
#    echo $output_file
#
#    python -u 009.010.2025-02-20-comb-cov-bin-mpm-quad.py ${lambda_idx} ${nbins} &> ${output_file}
#######################
#######################
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

python -u 009.010.2025-02-20-comb-cov-bin-mpm-quad.py ${lambda_idx} ${nbins}

EOF
    echo ${job_name}
#######################
done

echo "All jobs submitted"

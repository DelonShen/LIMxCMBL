#!/bin/bash
lambda_values=$(python3 -c '
import numpy as np
lambda_idxs = np.arange(25)[16:][::-1]
print("\n".join(map(str, lambda_idxs)))
')

readarray -t lambda_idxs <<< "$lambda_values"

date=$(date +%Y-%m-%d)
output_dir="logs"

nb=100


# Set the Slurm parameters
partition="kipac"
time_limit="24:00:00"
num_nodes=1
mem_per_node="3G"
cpus_per_task=1
output_dir="logs"

mkdir -p ${output_dir}

date=$(date +%Y-%m-%d)

for lambda_idx in "${lambda_idxs[@]}"; do
    echo $lambda_idx
    for curr in $(seq 0 $((${nb} - 1))); do
      lambda_formatted=$(echo $lambda_idx | tr '.' 'p')
      
      job_name="009.011-HETDEX-${lambda_formatted}-nb-${nb}-${curr}-dblquad"
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

python -u 009.011.2025-03-13.ILo-kappa-dbl-quadvec-Lya.py ${lambda_idx} ${nb} ${curr}

EOF
      echo ${job_name}

done

echo "Waiting 1 hour before processing next lambda_idx..."
sleep 3600
done

echo "All jobs submitted"

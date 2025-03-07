#!/bin/bash
lambda_values=$(python3 -c '
import numpy as np
lambda_idxs = np.arange(25)[12:][::-1]
print("\n".join(map(str, lambda_idxs)))
')

readarray -t lambda_idxs <<< "$lambda_values"

nb=100
nmc=30

date=$(date +%Y-%m-%d)

# Set the Slurm parameters
partition="owners"
time_limit="10:00:00"
num_nodes=1
mem_per_node="64G"
cpus_per_task=1

for bidx in $(seq 0 4 99); do
  for lambda_idx in "${lambda_idxs[@]}"; do
    lambda_formatted=$(echo $lambda_idx | tr '.' 'p')
    
    job_name="AtLAST-010.017-${lambda_formatted}-nb-${nb}-b1idx-${bidx}-nmc-${nmc}"
    output_file="logs/${date}-${job_name}.out"
    error_file="logs/${date}-${job_name}.err"

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

for a in \$(seq ${bidx} $((bidx+3))); do
  for b in \$(seq \${a} 99); do
    python -u 010.017.2025-03-05-comb-mc-sample-bin-specific.py ${lambda_idx} ${nb} ${nmc} \${a} \${b} 1 5
  done
done

EOF
    echo ${job_name}
  done
done

echo "All jobs submitted"

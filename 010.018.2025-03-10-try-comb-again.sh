#!/bin/bash
lambda_values=$(python3 -c '
import numpy as np
lambda_idxs = np.arange(25)[16:][::-1]
print("\n".join(map(str, lambda_idxs)))
')

zmin=5
zmax=8
next=1201

readarray -t lambda_idxs <<< "$lambda_values"

date=$(date +%Y-%m-%d)

# Set the Slurm parameters
partition="owners"
time_limit="12:00:00"
num_nodes=1
mem_per_node="64G"
cpus_per_task=3

for lambda_idx in "${lambda_idxs[@]}"; do
    lambda_formatted=$(echo $lambda_idx | tr '.' 'p')
    
    job_name="010.018-${lambda_formatted}-${zmin}-${zmax}-${next}"
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
#SBATCH -C GPU_MEM:80GB

for l in \$(seq 0 99); do
    python -u 010.018.2025-03-10-try-comb-again.py ${lambda_idx} ${next} \${l} ${zmin} ${zmax} Lya
done

EOF
    echo ${job_name}
done

echo "All jobs submitted"

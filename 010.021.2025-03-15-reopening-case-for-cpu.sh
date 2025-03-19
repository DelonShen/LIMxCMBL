#!/bin/bash
lambda_values=$(python3 -c '
import numpy as np
lambda_idxs = np.arange(25)[17:][::-1]
print("\n".join(map(str, lambda_idxs)))
')

readarray -t lambda_idxs <<< "$lambda_values"
next=301


date=$(date +%Y-%m-%d)

# Set the Slurm parameters
partition="kipac"
time_limit="72:00:00"
num_nodes=1
mem_per_node="32G"
cpus_per_task=1



input_file="LIMxCMBL/experiments.txt"

for lambda_idx in "${lambda_idxs[@]}"; do
  while IFS= read -r line; do
      if [ -z "$line" ]; then
          continue
      fi
      
      read -r name line zmin zmax <<< "$line"
      
    lambda_formatted=$(echo $lambda_idx | tr '.' 'p')
    
    job_name="010.021-${name}-${lambda_formatted}-${zmin}-${zmax}-${next}"
    output_file="logs/${date}-${job_name}.out"
    error_file="logs/${date}-${job_name}.err"

    sbatch << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${output_file}-%a
#SBATCH --error=${error_file}-%a
#SBATCH --time=${time_limit}
#SBATCH -p ${partition}
#SBATCH --nodes=${num_nodes}
#SBATCH --mem=${mem_per_node}
#SBATCH --cpus-per-task=${cpus_per_task}
#SBATCH --array=0-99

export JAX_PLATFORMS=cpu

python -u 010.021.2025-03-15-reopening-case-for-cpu.py ${lambda_idx} ${next} \${SLURM_ARRAY_TASK_ID} ${zmin} ${zmax} ${line}

EOF
    echo ${job_name}
#echo "Waiting 10 mins before processing next experiment..."
#sleep 600
done < "$input_file"
#echo "Waiting 1 hour before processing next lambda_idx..."
#sleep 3600
done

echo "All jobs submitted"


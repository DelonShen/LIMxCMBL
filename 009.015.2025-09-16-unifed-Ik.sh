#!/bin/bash
lambda_values=$(python3 -c '
import numpy as np
lambda_idxs = np.arange(25)[12:][::-1]
print("\n".join(map(str, lambda_idxs)))
')

readarray -t lambda_idxs <<< "$lambda_values"

date=$(date +%Y-%m-%d)
output_dir="logs"



# Set the Slurm parameters
partition="kipac"
time_limit="168:00:00"
num_nodes=1
mem_per_node="64G"
cpus_per_task=32
output_dir="logs"

mkdir -p ${output_dir}

date=$(date +%Y-%m-%d)


input_file="LIMxCMBL/experiments.txt"
while IFS= read -r line; do
      if [ -z "$line" ]; then
          continue
      fi
      
      read -r name line zmin zmax lm <<< "$line"
      echo ${name}
      echo ${zmin}
      echo ${zmax}
 
    job_name="009.015-${zmin}-${zmax}-${line}"
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


python -u 009.015.2025-09-16-unifed-Ik.py ${zmin} ${zmax} ${line} nLC
python -u 009.015.2025-09-16-unifed-Ik.py ${zmin} ${zmax} ${line}


EOF
    echo ${job_name}
#######################
done < "$input_file"


echo "All jobs submitted"

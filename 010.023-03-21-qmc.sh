#!/bin/bash
nb=100


date=$(date +%Y-%m-%d)

# Set the Slurm parameters
partition="kipac"
time_limit="72:00:00"
num_nodes=1
cpus_per_task=8
mem_per_node="32G"




input_file="LIMxCMBL/experiments.txt"
while IFS= read -r line; do
      if [ -z "$line" ]; then
          continue
      fi
      
      read -r name line zmin zmax lm <<< "$line"
      
for lambda_idx in $(seq 24 -1 ${lm}); do
    job_name="010.023-${name}-${lambda_idx}-${zmin}-${zmax}-${nb}"
    sbatch << EOF
#!/bin/bash

#SBATCH --job-name=${job_name}
#SBATCH --output="logs/${date}-${job_name}-%a.out"
#SBATCH --error="logs/${date}-${job_name}-%a.err"
#SBATCH --time=${time_limit}
#SBATCH -p ${partition}
#SBATCH --nodes=${num_nodes}
#SBATCH --mem=${mem_per_node}
#SBATCH --cpus-per-task=${cpus_per_task}
#SBATCH --array=0-10:10

export JAX_PLATFORMS=cpu

for i in {0..9}; do
  read a b <<< "\$(python 010.023-03-21-aux.py \$((\$SLURM_ARRAY_TASK_ID+i)))"
  python -u 010.023-03-21-qmc-cross.py ${lambda_idx} ${nb} \${a} \${b} ${zmin} ${zmax} ${line}
#  python -u 010.023-03-21-qmc-auto.py ${lambda_idx} ${nb}  \${a} \${b} ${zmin} ${zmax} ${line}
done

EOF
    echo ${job_name}
done
done < "$input_file"

echo "All jobs submitted"

#!/bin/bash
nb=100


date=$(date +%Y-%m-%d)

# Set the Slurm parameters
partition="hns,owners"
time_limit="2:00:00"
num_nodes=1
cpus_per_task=1
mem_per_node="8G"




input_file="LIMxCMBL/experiments.txt"
while IFS= read -r line; do
      if [ -z "$line" ]; then
          continue
      fi
      
      read -r name line zmin zmax lm <<< "$line"
      
for lambda_idx in $(seq 24 -1 ${lm}); do
    job_name="010.023-comb-${name}-${lambda_idx}-${zmin}-${zmax}-${nb}"
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
#SBATCH --array=0-5049:50
#SBATCH -G 1

for i in {0..49}; do
  read a b <<< "\$(python 010.023-03-21-aux.py \$((\$SLURM_ARRAY_TASK_ID+i)))"
  python -u 010.023-03-25-qmc-comb-gpu.py ${lambda_idx} ${nb} \${a} \${b} ${zmin} ${zmax} ${line}
done

EOF
    echo ${job_name}
    break
done
done < "$input_file"

echo "All jobs submitted"

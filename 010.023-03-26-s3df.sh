#!/bin/bash
nb=100

date=$(date +%Y-%m-%d)

# Set the Slurm parameters
partition="ampere"
time_limit="13:00:00"
num_nodes=1
mem_per_node="8G"




input_file="LIMxCMBL/experiments.txt"
while IFS= read -r line; do
      if [ -z "$line" ]; then
          continue
      fi
      
      read -r name line zmin zmax lm <<< "$line"
for lambda_idx in $(seq 24 -1 ${lm}); do
  for midx in $(seq 0 1010 5049); do
    job_name="010.023-comb-${name}-${lambda_idx}-${zmin}-${zmax}-${nb}-${midx}"
    sbatch << EOF
#!/bin/bash

#SBATCH --job-name=${job_name}
#SBATCH --account=kipac:kipac
#SBATCH --output="logs/${date}-${job_name}.out"
#SBATCH --error="logs/${date}-${job_name}.err"
#SBATCH --time=${time_limit}
#SBATCH --partition='${partition}'
#SBATCH --nodes=${num_nodes}
#SBATCH --mem=${mem_per_node}
#SBATCH --gpus 1

for i in {0..1009}; do
  read a b <<< "\$(python 010.023-03-21-aux.py \$((${midx}+i)))"
  python -u 010.023-03-25-qmc-comb-gpu.py ${lambda_idx} ${nb} \${a} \${b} ${zmin} ${zmax} ${line}
done

EOF
    echo ${job_name}
  done
done
done < "$input_file"

echo "All jobs submitted"

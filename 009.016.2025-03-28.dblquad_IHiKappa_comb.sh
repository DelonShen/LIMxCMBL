#!/bin/bash
nb=15

date=$(date +%Y-%m-%d)

# Set the Slurm parameters
partition="kipac"
time_limit="168:00:00"
num_nodes=1
#mem_per_node="3G"
#cpus_per_task=32
mem_per_node="1024G"
cpus_per_task=256

output_dir="logs"


input_file="LIMxCMBL/experiments.txt"
while IFS= read -r line; do
      if [ -z "$line" ]; then
          continue
      fi
      
      read -r name line zmin zmax lm <<< "$line"
      
#for lambda_idx in $(seq 24 -1 ${lm}); do
for lambda_idx in $(seq 23 23); do
asdf=3
      lambda_formatted=$(echo $lambda_idx | tr '.' 'p')
      
#      job_name="009.016-${name}-${lambda_idx}-nb-${nb}-dblquad"
      job_name="009.016-${name}-${lambda_idx}-nb-${nb}-${asdf}-dblquad"
      output_file="${output_dir}/${date}-${job_name}.out"
      error_file="${output_dir}/${date}-${job_name}.err"

    sbatch << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output="${output_dir}/${date}-${job_name}-${asdf}.out"
#SBATCH --error="${output_dir}/${date}-${job_name}-${asdf}.err"
#SBATCH --time=${time_limit}
#SBATCH -p ${partition}
#SBATCH --nodes=${num_nodes}
#SBATCH --mem=${mem_per_node}
#SBATCH --cpus-per-task=${cpus_per_task}

python -u 009.016.2025-03-28.dblquad_IHiKappa_comb.py ${lambda_idx} ${nb} ${asdf} ${zmin} ${zmax} ${line}

EOF
      echo ${job_name}
  done
done < "$input_file"

echo "All jobs submitted"

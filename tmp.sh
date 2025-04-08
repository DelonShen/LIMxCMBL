#!/bin/bash
nb=100

date=$(date +%Y-%m-%d)

# Set the Slurm parameters
partition="kipac"
time_limit="168:00:00"
num_nodes=1
mem_per_node="256G"
cpus_per_task=256
output_dir="logs"


input_file="LIMxCMBL/experiments.txt"
while IFS= read -r line; do
      if [ -z "$line" ]; then
          continue
      fi
      
      read -r name line zmin zmax lm <<< "$line"
      
for lambda_idx in $(seq 24 -1 ${lm}); do
      lambda_formatted=$(echo $lambda_idx | tr '.' 'p')
      
      job_name="009.016-${name}-${lambda_idx}-nb-${nb}-dblquad"
      output_file="${output_dir}/${date}-${job_name}.out"
      error_file="${output_dir}/${date}-${job_name}.err"

    sbatch << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output="${output_dir}/${date}-${job_name}-60.out"
#SBATCH --error="${output_dir}/${date}-${job_name}-60.err"
#SBATCH --time=${time_limit}
#SBATCH -p ${partition}
#SBATCH --nodes=${num_nodes}
#SBATCH --mem=${mem_per_node}
#SBATCH --cpus-per-task=${cpus_per_task}

python -u 009.016.2025-03-28.dblquad_IHiKappa_comb.py ${lambda_idx} ${nb} 60 ${zmin} ${zmax} ${line}

EOF
      echo ${job_name}
      break
  done
  break
done < "$input_file"

echo "All jobs submitted"

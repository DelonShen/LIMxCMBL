#!/bin/bash
date=$(date +%Y-%m-%d)

# Set the Slurm parameters
partition="kipac"
time_limit="168:00:00"
num_nodes=1
mem_per_node="8G"
cpus_per_task=1
output_dir="logs"


input_file="LIMxCMBL/experiments.txt"
while IFS= read -r line; do
      if [ -z "$line" ]; then
          continue
      fi
      
      read -r name line zmin zmax l0 l1 l2 l3 l4 <<< "$line"
      
for lambda_idx in $l0 $l1 $l2 $l3 $l4; do
      lambda_formatted=$(echo $lambda_idx | tr '.' 'p')
      
      job_name="009.017-${name}-${lambda_idx}-nb-${nb}-dblquad"
      output_file="${output_dir}/${date}-${job_name}.out"
      error_file="${output_dir}/${date}-${job_name}.err"

    sbatch << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output="${output_dir}/${date}-${job_name}-%a.out"
#SBATCH --error="${output_dir}/${date}-${job_name}-%a.err"
#SBATCH --time=${time_limit}
#SBATCH -p ${partition}
#SBATCH --nodes=${num_nodes}
#SBATCH --mem=${mem_per_node}
#SBATCH --cpus-per-task=${cpus_per_task}
#SBATCH --array=0-${nb}

python -u 009.017.2025-05-06.IHiKappa_Limber.py ${name} ${lambda_idx} \$SLURM_ARRAY_TASK_ID

EOF
      echo ${job_name}
      break
  done
  break
done < "$input_file"

echo "All jobs submitted"

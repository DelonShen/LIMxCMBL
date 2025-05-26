#!/bin/bash
nb=15

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
      
      read -r name line zmin zmax lm <<< "$line"
      
#for lambda_idx in $(seq 24 -1 ${lm}); do
for lambda_idx in $(seq 23 23); do
curr_bin=3
for ell_idx in $(seq 0 99); do
      job_name="009.016-${name}-${lambda_idx}-nb-${nb}-${curr_bin}-ell-idx-${ell_idx}-dblquad"
      output_file="${output_dir}/${date}-${job_name}.out"
      error_file="${output_dir}/${date}-${job_name}.err"

    sbatch << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output="${output_dir}/${date}-${job_name}-${curr_bin}.out"
#SBATCH --error="${output_dir}/${date}-${job_name}-${curr_bin}.err"
#SBATCH --time=${time_limit}
#SBATCH -p ${partition}
#SBATCH --nodes=${num_nodes}
#SBATCH --mem=${mem_per_node}
#SBATCH --cpus-per-task=${cpus_per_task}

python -u 009.016.2025-05-25.dblquad_IHiKappa_comb_individual_ell.py ${lambda_idx} ${nb} ${curr_bin} ${zmin} ${zmax} ${line} ${ell_idx}

EOF
      echo ${job_name}
    done
  done
done < "$input_file"

echo "All jobs submitted"

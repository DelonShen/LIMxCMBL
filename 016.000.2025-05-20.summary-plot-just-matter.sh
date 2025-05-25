#!/bin/bash
date=$(date +%Y-%m-%d)

# Set the Slurm parameters
partition="kipac"
time_limit="48:00:00"
num_nodes=1
mem_per_node="1024G"
cpus_per_task=1

output_dir="logs"

nm=256

job_name="016.000-${nm}"
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

python -u 016.000.2025-05-20.summary-plot-just-matter.py ${nm}

EOF

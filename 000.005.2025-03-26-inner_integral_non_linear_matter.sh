#!/bin/bash

#for curr in {0..99}; do
for curr in "5" "13"; do
    echo $curr
    # Set the Slurm job parameters
    job_name_prefix="nonlinearmatter-inner-integral-ell-idx-${curr}"
    output_dir="logs"
    time_limit="168:00:00"
    partition="kipac"
    num_nodes=1
    mem_per_node="64G"
    cpus_per_task=256
    
    # Create a Slurm job script for the data file
    job_name="${job_name_prefix}"
    job_script="scripts/${job_name}.sh"
    output_file="${output_dir}/$(date +%Y-%m-%d)-${job_name}.out"
    error_file="${output_dir}/$(date +%Y-%m-%d)-${job_name}.err"
    
    # Submit the job using a here document
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

python -u 000.005.2025-03-26-inner_integral_non_linear_matter.py ${curr}
EOF

done

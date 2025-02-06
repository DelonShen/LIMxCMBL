#!/bin/bash
lambda_values=$(python3 -c '
import numpy as np
lambdas = np.logspace(-5, 0, 50)
print("\n".join(map(str, lambdas)))
')

readarray -t lambdas <<< "$lambda_values"

# Set the Slurm parameters
partition="owners"
time_limit="24:00:00"
num_nodes=1
mem_per_node="64G"
cpus_per_task=1
output_dir="logs"

mkdir -p ${output_dir}

date=$(date +%Y-%m-%d)

for lambda in "${lambdas[@]}"; do
    echo $lambda
    lambda_formatted=$(echo $lambda | tr '.' 'p')
    
    job_name="IHi-kappa-no-window-Lambda-${lambda_formatted}"
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

python -u 002.009.2025-02-05.compute_IHi_Kappa_no_window.py ${lambda}

          
EOF

    echo "Submitted job for Lambda = ${lambda}"
done

echo "All jobs submitted"

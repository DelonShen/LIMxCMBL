#!/bin/bash
date=$(date +%Y-%m-%d)

#really should be
#SBATCH --array=0-$(( total_jobs - 1 ))

# Set the Slurm parameters
partition="kipac"
time_limit="60:00"
num_nodes=1
mem_per_node="32G"
cpus_per_task=8
output_dir="logs"

mx=49
total_jobs=$(( (mx + 1) * (mx + 1) ))

#have not run 901-1800 yet
#have not run 1801-rest yet

sbatch << EOF
#!/bin/bash
#SBATCH --job-name=008.023-grid
#SBATCH --output="${output_dir}/${date}-008.023-%a.out"
#SBATCH --error="${output_dir}/${date}-008.023-%a.err"
#SBATCH --time=${time_limit}
#SBATCH -p ${partition}
#SBATCH --nodes=${num_nodes}
#SBATCH --mem=${mem_per_node}
#SBATCH --cpus-per-task=${cpus_per_task}
#SBATCH --array=1801-$(( total_jobs - 1 ))


# Calculate a0 and b0 from the array task ID
mx=${mx}
a0=\$(( \${SLURM_ARRAY_TASK_ID} / (\${mx} + 1) ))
b0=\$(( \${SLURM_ARRAY_TASK_ID} % (\${mx} + 1) ))

echo "Running job for a0=\${a0}, b0=\${b0} (array task \${SLURM_ARRAY_TASK_ID})"

ml mathematica
cd ~/LIMxCMBL
./008.023.2025-09-12-analytical-eLOeLO-DFT-summand.wls \${a0} \${b0}

EOF

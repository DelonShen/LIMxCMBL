#!/bin/bash
acc="kipac:default"
partition="ampere"
ngpu=4
time_limit="168:00:00"

mem_per_node="64G"




date=$(date +%Y-%m-%d)
nb=15


input_file="LIMxCMBL/experiments.txt"
while IFS= read -r line; do
      if [ -z "$line" ]; then
          continue
      fi
      
      read -r name line zmin zmax lm <<< "$line"
#for lambda_idx in $(seq ${lm} 24); do
for lambda_idx in $(seq 24 -1 ${lm}); do
#  for midx in $(seq 0 1010 5049); do
  for midx in $(seq 0 40 119); do
    job_name="${acc}-${partition}-010.023-comb-${name}-${lambda_idx}-${zmin}-${zmax}-${nb}-${midx}"
    sbatch << EOF
#!/bin/bash

#SBATCH --job-name=${job_name}
#SBATCH --account=${acc}
#SBATCH --output="logs/${date}-${job_name}.out"
#SBATCH --error="logs/${date}-${job_name}.err"
#SBATCH --time=${time_limit}
#SBATCH --partition='${partition}'
#SBATCH --nodes=1
#SBATCH --mem=${mem_per_node}
#SBATCH --gpus ${ngpu}
#SBATCH --cpus-per-gpu=1

run_task_on_gpu() {
    local gpu_id=\$1
    local task_args="\${@:2}"
    CUDA_VISIBLE_DEVICES=\$gpu_id python -u 010.023-03-25-qmc-comb-gpu.py \${task_args} &
}

declare -a pids=()

for i in {0..39}; do
  read a b <<< "\$(python 010.023-03-21-aux.py \$((${midx}+i)) ${nb})"
  gpu_index=\$(((${midx}+i) % ${ngpu}))

  echo ${lambda_idx} ${nb} \${a} \${b} ${zmin} ${zmax} ${line}
  run_task_on_gpu \$gpu_index ${lambda_idx} ${nb} \${a} \${b} ${zmin} ${zmax} ${line}

  pids+=(\$!)
  if (( ((${midx}+i) + 1) % ${ngpu} == 0 )); then
      for pid in "\${pids[@]}"; do
          wait \$pid
      done
      pids=()
  fi

done

for pid in "\${pids[@]}"; do
    wait \$pid
done

EOF
    echo ${job_name}
  done
done
done < "$input_file"

echo "All jobs submitted"

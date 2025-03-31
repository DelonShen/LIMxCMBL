#!/bin/bash
acc="kipac:kipac"
partition="ada"
ngpu=5
time_limit="240:00:00"
#wait or no?

mem_per_node="32G"




date=$(date +%Y-%m-%d)
nb=100


input_file="LIMxCMBL/experiments.txt"
while IFS= read -r line; do
      if [ -z "$line" ]; then
          continue
      fi
      read -r name line zmin zmax lm <<< "$line"

    job_name="${acc}-${partition}-010.023-comb-${name}-${zmin}-${zmax}-${nb}"
    sbatch << EOF
#!/bin/bash

#SBATCH --requeue
#SBATCH --job-name=${job_name}
#SBATCH --account=${acc}
#SBATCH --output="logs/${date}-${job_name}.out"
#SBATCH --error="logs/${date}-${job_name}.err"
#SBATCH --time=${time_limit}
#SBATCH --partition='${partition}'
#SBATCH --mem=${mem_per_node}
#SBATCH --nodes=1
#SBATCH --gpus ${ngpu}
#SBATCH --cpus-per-gpu=1

run_task_on_gpu() {
    local gpu_id=\$1
    local task_args="\${@:2}"
    CUDA_VISIBLE_DEVICES=\$gpu_id python -u 010.023-03-25-qmc-comb-gpu.py \${task_args} &
}

declare -a pids=()

for lambda_idx in \$(seq 24 -1 ${lm}); do
  for i in \$(seq 0 5049); do
      read a b <<< "\$(python 010.023-03-21-aux.py \$((i)))"
      gpu_index=\$((i % ${ngpu}))
      run_task_on_gpu \$gpu_index \${lambda_idx} ${nb} \${a} \${b} ${zmin} ${zmax} ${line}
      
      pids+=(\$!)
      
      if (( (i + 1) % ${ngpu} == 0 )); then
          for pid in "\${pids[@]}"; do
              wait \$pid
          done
          pids=()
      fi
  done
  
  for pid in "\${pids[@]}"; do
      wait \$pid
  done
done

EOF
    echo ${job_name}
done < "$input_file"

echo "All jobs submitted"

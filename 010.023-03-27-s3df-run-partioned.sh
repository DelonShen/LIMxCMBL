#!/bin/bash
#nb=100
nb=15 #SPHEREx

#mli:defualt has turing (? gpu) or ampere (4 gpu)
#kipac:default ampere (4 gpu)
#kipac:kipac ada (5 gpu)
#mli:cmb-ml has turing (10 gpu) or ampere (4gpu)
#but I think turing GPU not enough memory

#ngpu=10 #turing
ngpu=4 #ampere

acc="mli:cmb-ml"
partition="ampere"
time_limit="168:00:00"

mem_per_node="64G"




date=$(date +%Y-%m-%d)



input_file="LIMxCMBL/experiments.txt"
while IFS= read -r line; do
      if [ -z "$line" ]; then
          continue
      fi
      read -r name line zmin zmax l0 l1 l2 l3 l4 <<< "$line"



for lambda_idx in $l4 $l3 $l2 $l1 $l0; do
#  step=101
  step=40 #for 15 bins

#  for midx in $(seq 0 ${step} 5049); do # for 100 bins
#  for midx in $(seq 0 51 1274); do # for 50 bins
  for midx in $(seq 0 ${step} 119); do # for 15 bins
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

for i in {0..${step}}; do
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

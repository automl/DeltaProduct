#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=length_eval
#SBATCH --partition=<cluster specific partition>
#SBATCH --time=<cluster specific time limit>
#SBATCH --output=<path to slurm logs>/%A_%a_%N_log.out
#SBATCH --error=<path to slurm logs>/%A_%a_%N_log.err
#SBATCH --array=0-75
#SBATCH --mail-type=FAIL

project_home=<path to project home>
cd $project_home || exit 1
source .venv/bin/activate || exit 1
export PYTHONPATH=$PYTHONPATH:$project_home

datasets=(slimpajama_6b trivia_qa codeparrot openthoughts-114k-math)
paths=(
    "${project_home}/legacy/training/runs/DN_-1_1_2k/checkpoint-66758" \
    "${project_home}/legacy/training/runs/DN_-1_1_4k/checkpoint-66758" \
    "${project_home}/legacy/training/runs/DN_-1_1_4k_12h/checkpoint-66758" \
    "${project_home}/legacy/training/runs/DN_-1_1_4k_213M/checkpoint-36240" \ 
    "${project_home}/legacy/training/runs/DN_805M/checkpoint-104905" \
    "${project_home}/legacy/training/runs/DP2_-1_1_4k_8h/checkpoint-66758" \
    "${project_home}/legacy/training/runs/DP2_-1_1_4k_213M/checkpoint-36240" \
    "${project_home}/legacy/training/runs/DP2_805M/checkpoint-104905" \
    "${project_home}/legacy/training/runs/DP3_-1_1_2k/checkpoint-66758" \
    "${project_home}/legacy/training/runs/DP3_-1_1_4k/checkpoint-66758" \
    "${project_home}/legacy/training/runs/DP3_-1_1_4k_6h/checkpoint-66758" \
    "${project_home}/legacy/training/runs/DP3_-1_1_4k_213M/checkpoint-36240" \
    "${project_home}/legacy/training/runs/DP3_-1_1_8k/checkpoint-66758" \
    "${project_home}/legacy/training/runs/DP3_64head_dim/checkpoint-66758" \
    "${project_home}legacy/training/runs/DP3_805M/checkpoint-104905" \
    "${project_home}/legacy/training/runs/GDN_-1_1_2k/checkpoint-66758" \
    "${project_home}/legacy/training/runs/GDN_-1_1_4k/checkpoint-66758" \
    "${project_home}/legacy/training/runs/GDP2_-1_1_4k/checkpoint-66758" \
    "${project_home}/legacy/training/runs/GDP3_-1_1_4k/checkpoint-66758"
)

# Extract model names from paths
model_names=()
for p in "${paths[@]}"; do
    temp="${p%/*}" # Remove /checkpoint-XXXX
    model_names+=("${temp##*/}") # Extract model name
done

max_len=32768

# Array job: for each (dataset, model) pair
idx=${SLURM_ARRAY_TASK_ID}

if [ -z "$idx" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID not set."
    exit 1
fi

num_datasets=${#datasets[@]}
num_models=${#paths[@]}
total_jobs=$((num_datasets * num_models))

if [ "$idx" -ge "$total_jobs" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($idx) exceeds total jobs ($total_jobs)"
    exit 1
fi

dataset_idx=$((idx / num_models))
model_idx=$((idx % num_models))

dataset="${datasets[$dataset_idx]}"
checkpoint="${paths[$model_idx]}"
model_name="${model_names[$model_idx]}"

echo "Running length extrapolation for model: $model_name, checkpoint: $checkpoint, dataset: $dataset, max_len: $max_len"
srun python custom_evals/length_extrapolation.py \
    --path "$checkpoint" \
    --max_len "$max_len" \
    --data "$dataset" \
    --model_name "$model_name" \
    --batch_size 2 \
    --num_cpu_workers 8

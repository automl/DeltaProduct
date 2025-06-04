#!/bin/bash
#SBATCH --job-name=effective_rank
#SBATCH --output=<path to slurm logs>/effective_rank_%A_%a.out
#SBATCH --error=<path to slurm logs>/effective_rank_%A_%a.err
#SBATCH --array=0-23
#SBATCH --time=<cluster specific time limit>
#SBATCH --partition=<cluster specific partition>
#SBATCH --gres=gpu:1

# Create logs directory if it doesn't exist

# Define datasets and layers to iterate over
datasets=("trivia_qa" "codeparrot" "gsm8k" "deepmind_math_dataset")
layers=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23)

dataset_idx=$((SLURM_ARRAY_TASK_ID / ${#layers[@]}))
layer_idx=$((SLURM_ARRAY_TASK_ID % ${#layers[@]}))

# Get the actual dataset and layer values
dataset=${datasets[$dataset_idx]}
layer=${layers[$layer_idx]}

# Fixed sequence length (modify if needed)
seq_len=16384

echo "Running effective rank analysis for dataset=$dataset, layer=$layer, seq_len=$seq_len"

project_home=<path to project home>

# Activate your environment if needed
source $project_home/.venv/bin/activate
export PYTHONPATH="$project_home:$PYTHONPATH"

# Run the Python script
python ${project_home}/custom_evals/effective_state_rank.py \
    --dataset $dataset \
    --layer $layer \
    --seq_len $seq_len \
    --model_name DP3_64head_dim \
    --num_householder 3 \
    --train_context_len 4096
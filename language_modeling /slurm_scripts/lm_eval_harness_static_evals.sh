#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=<job name>
#SBATCH --partition=<cluster specific partition>
#SBATCH --time=<cluster specific time limit>
#SBATCH --output=<path to slurm logs>/%j_%a_%N_log.out
#SBATCH --error=<path to slurm logs>/%j_%a_%N_log.err
#SBATCH --array=0-8

model_list=(
    DN_-1_1_4k_213M/checkpoint-36240 \
    DP2_-1_1_4k_213M/checkpoint-36240 \
    DP3_-1_1_4k_213M/checkpoint-36240 \
    DN_-1_1_4k_12h/checkpoint-66758 \
    DP2_-1_1_4k_8h/checkpoint-66758 \
    DP3_-1_1_4k_6h/checkpoint-66758 \
    DN_805M/checkpoint-104905 \
    DP2_805M/checkpoint-104905 \
    DP3_805M/checkpoint-104905 \
)

base_path=<path to runs directory>

# Get the model for this array task
model=${model_list[$SLURM_ARRAY_TASK_ID]}
checkpoint=$base_path/$model

echo "Evaluating $checkpoint"

project_home=<path to project home>
cd $project_home
source .venv/bin/activate

export PYTHONPATH=$PYTHONPATH:$project_home
export CUDA_LAUNCH_BLOCKING=1

srun python -m evals.harness \
--model fla \
--model_args pretrained=$checkpoint,dtype=bfloat16 \
--batch_size 64 \
--tasks wikitext,lambada,piqa,hellaswag,winogrande,arc_easy,arc_challenge,swde,squad_completion,fda \
--num_fewshot 0 \
--device cuda \
--show_config \
--output_path=<path to output directory> \
--trust_remote_code \
--seed 42

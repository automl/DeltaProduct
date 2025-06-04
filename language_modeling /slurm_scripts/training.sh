#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=<job name>
#SBATCH --partition=<cluster specific partition>
#SBATCH --time=<cluster specific time limit>
#SBATCH --output=<path to slurm logs>/%j_%a_%N_log.out
#SBATCH --error=<path to slurm logs>/%j_%a_%N_log.err
#SBATCH --mail-type=BEGIN,END,FAIL

echo GPUs per node: $SLURM_GPUS_ON_NODE

project_home=<path to project home>
data_home=<path to tokenized and chunked data>
log_path=${project_home}/legacy/training/runs/<log path>

cd ${project_home}
source .venv/bin/activate
cd training

# ml load devel/cuda/12.4  # Cluster specific

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32
export NCCL_DEBUG=INFO
export NCCL_MIN_NCHANNELS=11 
export NCCL_TREE_THRESHOLD=4294967296 
export TORCH_DISTRIBUTED_DEBUG=INFO

# tuning OMP_NUM_THREADS
export OMP_NUM_THREADS=8


export RDZV_HOST=$(hostname)
export RDZV_PORT=29400

# export WANDB_RESUME=allow
export WANDB_NAME="$type.$(basename $log_path)"
export WANDB_PROJECT=<wandb project name>
export WANDB_ENTITY=<wandb entity name>
export WANDB_RUN_ID="$WANDB_NAME-$date"


srun ../.venv/bin/torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
    run.py \
    --model_name_or_path ${project_home}/legacy/training/configs/gated_multi_deltanet_340m_-1_1_3.json \
    --tokenizer mistralai/Mistral-7B-v0.1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --use_fast_tokenizer \
    --do_train \
    --dataset HuggingFaceFW/fineweb \
    --context_length 2048 \
    --streaming \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 8 \
    --dataloader_prefetch_factor 3 \
    --ignore_data_skip \
    --output_dir $log_path \
    --overwrite_output_dir \
    --logging_steps 32 \
    --include_num_input_tokens_seen \
    --save_steps 2048 \
    --save_total_limit 16 \
    --learning_rate 3e-4 \
    --lr_scheduler_type cosine_with_min_lr \
    --warmup_steps 512 \
    --optim adamw_torch_fused \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_grad_norm 1.0 \
    --num_train_epochs 1 \
    --seed 42 \
    --logging_steps 32 \
    --bf16 \
    --torch_compile False \
    --max_steps 66758 \
    --cache_dir ${data_home} \
    --deepspeed ${project_home}/training/configs/deepspeed_config.json
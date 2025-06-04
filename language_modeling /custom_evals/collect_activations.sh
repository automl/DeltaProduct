#!/bin/bash

# Define the datasets in an array
datasets=("trivia_qa" "gsm8k" "codeparrot" "deepmind_math_dataset")

path="<path to checkpoint>"
# Loop over each dataset and execute the command
for dataset in "${datasets[@]}"
do
    PYTHONPATH=<path to project home> python custom_evals/collect_activations.py --data $dataset --model_name DP3_64head_dim --path $path
done

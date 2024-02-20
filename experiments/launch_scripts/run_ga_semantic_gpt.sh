#!/bin/bash

#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export model=$1 # llama2 or vicuna
export setup=$2 # behaviors or strings

# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi
# data_offset for starting from a certain point 
for data_offset in 0
do

    python -u ../develop.py \
        --config="../configs/individual_${model}.py" \
        --config.attack=ga \
        --config.train_data="../../data/advbench/harmful_${setup}.csv" \
        --config.result_prefix="../results/attack_on_${model}" \
        --config.n_train_data=200 \
        --config.data_offset=$data_offset \
        --config.n_steps=1000 \
        --config.test_steps=50 \
        --config.batch_size=512 \
        --config.sentence_tokenizer=True \
        --config.algo=GA_semantic_advanced \
        --config.noun_sub=True \
        --config.verb_sub=True \
        --config.noun_wordgame=True \
        --config.suffix=True \

done
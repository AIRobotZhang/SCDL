#!/bin/bash
GPUID=$1
echo "Run on GPU $GPUID"

# data
DATASET=$2
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")
DATA_ROOT=$PROJECT_ROOT/dataset/

# model
TOKENIZER_TYPE=roberta
STUDENT1_TYPE=roberta
STUDENT2_TYPE=distilroberta
TOKENIZER_NAME=roberta-base
STUDENT1_MODEL_NAME=roberta-base
STUDENT2_MODEL_NAME=distilroberta-base

# self-collaborative learning parameters
LR=1e-5 
WARMUP=200 # Sche. Warmup
BEGIN_EPOCH=1 # Pre. Epoch
PERIOD=6000 # Update Cycle (iterations)
MEAN_ALPHA=0.995 # EMA α
THRESHOLD=0.9 # Confidence Threshold δ
TRAIN_BATCH=16 # 
EPOCH=50
LABEL_MODE=soft # Twitter hard


WEIGHT_DECAY=1e-4
SEED=0
ADAM_EPS=1e-8
ADAM_BETA1=0.9
ADAM_BETA2=0.98

EVAL_BATCH=32


# output
OUTPUT=$PROJECT_ROOT/ptms/$DATASET/

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUID python3 -u run_script.py --data_dir $DATA_ROOT \
  --student1_model_name_or_path $STUDENT1_MODEL_NAME \
  --student2_model_name_or_path $STUDENT2_MODEL_NAME \
  --output_dir $OUTPUT \
  --tokenizer_name $TOKENIZER_NAME \
  --cache_dir $PROJECT_ROOT/cached_models \
  --max_seq_length 128 \
  --learning_rate $LR \
  --weight_decay $WEIGHT_DECAY \
  --adam_epsilon $ADAM_EPS \
  --adam_beta1 $ADAM_BETA1 \
  --adam_beta2 $ADAM_BETA2 \
  --max_grad_norm 1.0 \
  --num_train_epochs $EPOCH \
  --warmup_steps $WARMUP \
  --per_gpu_train_batch_size $TRAIN_BATCH \
  --per_gpu_eval_batch_size $EVAL_BATCH \
  --gradient_accumulation_steps 1 \
  --logging_steps 100 \
  --save_steps 100000 \
  --evaluate_during_training \
  --seed $SEED \
  --overwrite_output_dir \
  --mean_alpha $MEAN_ALPHA \
  --self_learning_label_mode $LABEL_MODE \
  --self_learning_period $PERIOD \
  --model_type $TOKENIZER_TYPE \
  --begin_epoch $BEGIN_EPOCH \
  --do_train \
  --dataset $DATASET \
  --threshold $THRESHOLD \


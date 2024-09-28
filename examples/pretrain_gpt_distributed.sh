#!/bin/bash

# Runs the "175B" parameter model
export HOME=/mnt/hwfile/chenqiaoling
LLMPLATFORM=/mnt/hwfile/share_data/llm_env
export GCC_HOME=${LLMPLATFORM}/dep/gcc-10.2.0
export MPFR_HOME=${LLMPLATFORM}/dep/mpfr-4.1.0
export CUDA_PATH=${LLMPLATFORM}/dep/cuda-11.8
export CUDA_HOME=${CUDA_PATH}

export LD_LIBRARY_PATH=${GCC_HOME}/lib64
export LD_LIBRARY_PATH=${MPFR_HOME}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64/:$LD_LIBRARY_PATH
export CC=${GCC_HOME}/bin/gcc
export CXX=${GCC_HOME}/bin/c++

# export PATH=/usr/bin:/usr/local/sbin:/usr/sbin
export PATH=${GCC_HOME}/bin/:$PATH
export PATH=${CUDA_HOME}/bin/:$PATH

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_TIMEOUT=19
export NCCL_IB_QPS_PER_CONNECTION=4

export CPATH=/mnt/hwfile/share_data/llm_env/cuda-11.8/targets/x86_64-linux/include:$CPATH

export CPLUS_INCLUDE_PATH=${CUDA_HOME}/include:$CPLUS_INCLUDE_PATH

splits=$1
# __doc_head_address_start__
# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}

head_node_ip=$(cat /etc/hosts | grep -w "$head_node" | awk '{print $1}')
echo $head_node

## env config
GPUS_PER_NODE=2
# HOST-10-140-60-[33-70,82-84,86,88-91,94-97,102-106,108-109,113,124-129]
MASTER_ADDR=$head_node_ip
MASTER_PORT=7880
NNODES=$SLURM_NNODES

CHECKPOINT_PATH=./tmp #<Specify path>
TENSORBOARD_LOGS_PATH=./tmp #<Specify path>
# VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
# MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt
# DATA_PATH=$5 #<Specify path and file prefix>_text_document

VOCAB_FILE=/mnt/petrelfs/chenqiaoling/Seq1F1B/data/gpt2-vocab.json
MERGE_FILE=/mnt/petrelfs/chenqiaoling/Seq1F1B/data/gpt2-merges.txt
DATA_PATH=/mnt/petrelfs/chenqiaoling/Seq1F1B/data/meg-gpt2-oscar-en-10k_text_document


TENSOR_MODEL_PARALLEL_SIZE=1
NUM_LAYERS=2
HIDDEN_SIZE=4096
NUM_ATTENTION_HEADS=16
GLOBAL_BATCH_SIZE=1
SEQ_LEN=4096
TRAIN_SAMPLES=73242188  # 300B tokens / 4096
LR_WARMUP_SAMPLES=50000
LR_DECAY_SAMPLES=73192188 # TRAIN_SAMPLES - LR_WARMUP_SAMPLES
    #    --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE} \
        #    --sequence-parallel \
            #    --untie-embeddings-and-output-weights \
CHECKPOINT_DIR="./checkpoints"
DATACACHE_DIR="./data-cache"
TENSORBOARD_DIR="./tensorboard"
options=" \
       --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE} \
       --sequence-parallel \
        --data-path $DATA_PATH \
        --vocab-file $VOCAB_FILE \
        --merge-file $MERGE_FILE \
       --pipeline-model-parallel-size 2 \
       --untie-embeddings-and-output-weights \
       --use-flash-attn \
       --use-distributed-optimizer \
       --kaimm-offload-activation-ratio 1 \
       --untie-embeddings-and-output-weights \
       --init-method-std 0.02 \
       --position-embedding-type rope \
       --num-layers ${NUM_LAYERS} \
       --hidden-size ${HIDDEN_SIZE} \
       --num-attention-heads ${NUM_ATTENTION_HEADS} \
       --group-query-attention \
       --num-query-groups 8 \
       --seq-length ${SEQ_LEN} \
       --max-position-embeddings ${SEQ_LEN} \
       --train-samples ${TRAIN_SAMPLES} \
       --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
       --lr-decay-samples ${LR_DECAY_SAMPLES} \
       --split 99,1,0 \
       --tokenizer-type GPT2BPETokenizer \
       --distributed-backend nccl \
       --micro-batch-size 1 \
       --global-batch-size ${GLOBAL_BATCH_SIZE} \
       --swiglu \
       --lr 2.5e-4 \
       --min-lr 2.5e-5 \
       --lr-decay-style cosine \
       --weight-decay 0.1 \
       --clip-grad 1.0 \
       --attention-dropout 0.0 \
       --hidden-dropout 0.0 \
       --disable-bias-linear \
       --normalization RMSNorm \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 10 \
       --save-interval 2000 \
       --eval-interval 2000 \
       --eval-iters 32 \
       --bf16 \
       --pipe-sp-splits 2 \
       --tensorboard-dir ${TENSORBOARD_DIR}"
    #    --pipe-sp-splits 64 \

# conda activate /mnt/petrelfs/share_data/chenxun.p/envs/llm-torch2.1-flash2-nemo/ 

/mnt/petrelfs/share_data/chenxun.p/envs/llm-torch2.1-flash2-nemo/bin/torchrun --nproc_per_node 2 --nnodes $NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
/mnt/petrelfs/chenqiaoling/Seq1F1B/pretrain_gpt.py ${options}

#!/bin/bash

# set up the Data and checkpoint locations
OUTDIR=/lustre/out && mkdir -p $OUTDIR


# set up environment variables for Torch DistributedDataParallel
WORLD_SIZE_JOB=$SLURM_NTASKS
RANK_NODE=$SLURM_NODEID
PROC_PER_NODE=8
MASTER_ADDR_JOB=(`scontrol show hostnames $SLURM_JOB_NODELIST`)
MASTER_PORT_JOB="12234"

. /etc/parallelcluster/cfnconfig
HOME_DIR=/home/$cfn_cluster_user

IMDS_TOKEN=`curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"`
INSTANCEID=`curl -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" -v http://169.254.169.254/latest/meta-data/instance-id`
HOST=`hostname`

echo "Hostname: $HOST (instance ID: $INSTANCEID)"
echo "SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "MASTER_ADDR_JOB: $MASTER_ADDR_JOB"
echo "WORLD_SIZE_JOB: $WORLD_SIZE_JOB"
echo "RANK_NODE: $RANK_NODE"
echo "HOME_DIR: $HOME_DIR"

# checking Python version
PYTHON_VERSION=$(python --version 2>&1)
echo "Python version: $PYTHON_VERSION" 

# setup NCCL to use EFA
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0
export NCCL_SOCKET_IFNAME=eth0
export LD_LIBRARY_PATH=$HOME_DIR/nccl/build/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:$LD_LIBRARY_PATH

# set up HuggingFace token
HF_TOKEN=<huggingface-token>

torchrun \
    --nproc_per_node=$PROC_PER_NODE \
    --nnodes=$WORLD_SIZE_JOB \
    --node_rank=$RANK_NODE \
    --master_addr=${MASTER_ADDR_JOB} \
    --master_port=${MASTER_PORT_JOB} \
    /lustre/run_mlm.py \
    --model_config_id bert-base-uncased  \
    --dataset_id delmeng/processed_bert_dataset \
    --tokenizer_id delmeng/bert-base-uncased-2023-pc-p4d \
    --repository_id bert-base-uncased-2023 \
    --hf_hub_token $HF_TOKEN \
    --max_steps 1_000 \
    --per_device_train_batch_size 32 \
    --learning_rate 5e-5 | tee $OUTDIR/train.$RANK_NODE.$WORLD_SIZE_JOB.log
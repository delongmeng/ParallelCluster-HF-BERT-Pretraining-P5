## Distributed Large Language Model Pre-training Using ParallelCluster and AWS EC2 P5 Instances

### Model and Dataset

In this repo we show how to train the BERT model from scratch in a distributed environment. The original BERT was pretrained on Wikipedia and BookCorpus dataset. Both datasets are available on the Hugging Face Hub and can be loaded with datasets: [Wikipedia](https://huggingface.co/datasets/wikipedia), [BookCorpus](https://huggingface.co/datasets/bookcorpus).

Dataset preprocessing can be found in the `notebook/BERT-dataset-preprocessing.ipynb` Jupyter notebook. The processed dataset can be saved to a S3 bucket and used later for model pre-training. Compute resource: AWS EC2 c5n.18xlarge instance. The processed dataset can also be found [here](https://huggingface.co/datasets/delmeng/processed_bert_dataset) and we will directly use it (by providing HuggingFace dataset_id) in the pre-training step.


### Create a Cluster Using ParallelCluster

[AWS ParallelCluster](https://aws.amazon.com/hpc/parallelcluster/) is an AWS supported open source cluster management tool that helps you to deploy and manage high performance computing (HPC) clusters in the AWS Cloud. 

- ParallelCluster can be installed following the instruction [here](https://docs.aws.amazon.com/parallelcluster/latest/ug/install-v3-virtual-environment.html).  

- A custom AMI can be built for ParallelCluster. See the instruction [here](https://docs.aws.amazon.com/parallelcluster/latest/ug/building-custom-ami-v3.html). In this repo we use a custom ParallelCluster AMI built based on AWS Deep Learning AMI ([DLAMI](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html)).  

- A cluster can be configured and created following this [instruction](https://docs.aws.amazon.com/parallelcluster/latest/ug/install-v3-configuring.html). Here the configuration file `config/config_ml_p5_us-east-1f.yaml` was used to create a cluster using the following CLI command:

```
pcluster create-cluster --cluster-name myCluster --cluster-configuration config/config_ml_p5_us-east-1f.yaml
```

- Instances: This cluster has one head node with the instance type of c5n.2xlarge and a compute queue with p5.48xlarge instances, containing one so-called static node (always on) and one so-called dynamic node (only gets turned on when needed and automatically turned off when the job is finished). 

- Storage: Amazon FSx for Lustre file system is used as the storage solution, linked to a S3 bucket, and mounted to the `/lustre` path in the instance.

- Bootstrap configuration: A OnNodeConfigured script (see `scripts/setup.sh`), stored in a S3 bucket, is used as a "custom action" in the bootstrap process. This script will be executed at the end of the instance bootstrap process to set up the environment needed for the model training, including:

  - Setting up Python virtual environment and install Pytorch and other packages in the head node  
  - Setting up NCCL and EFA in compute nodes  

- SSM (or ssh) into the head node of the cluster. You can run NCCL test to make sure it's correctly set up.
```
sudo su - ubuntu

cd ~
git clone https://github.com/NVIDIA/nccl-tests.git
cd  nccl-tests/
make NCCL_HOME=$CUDA_DIRECTORY
NCCL_DEBUG=INFO build/all_reduce_perf -b 8 -f 2 -e 32M -c 1
```

- In the head node, you can try simple Slurm commands.

```
sinfo
squeue
srun -N 1 hostname
srun -N 2 hostname
sbatch -N 1 --wrap "sleep 10"
sbatch -N 2 --wrap "sleep 10"
scontrol show job --details
```


### Pre-train the BERT Model

- Activate the virtual environment, and prepare the directory.

```
source /home/ubuntu/ml_venv/bin/activate
cd /lustre 
mkdir p5_BERT_pretrain
```

Note that since this `/lustre` directory actually points to the S3 bucket that is linked to the FSx for Lustre file system, you have access to the files stored in this bucket, which can be a convenient way to prepare dataset and store training artifacts.


- On a single compute node, one can directly execute the `run_mlm.py` script (modified based on this [example](https://github.com/philschmid/deep-learning-habana-huggingface/blob/1a2022823b8899dad161b448e7cccc4a4e944fdf/pre-training/scripts/run_mlm.py), and can be found here `scripts/run_mlm.py` in this repo) from within the compute node as shown below:
```
python /lustre/run_mlm.py \
--model_config_id bert-base-uncased \
--dataset_id delmeng/processed_bert_dataset \
--tokenizer_id delmeng/bert-base-uncased-2023-pc-p5 \
--repository_id bert-base-uncased-2023 \
--hf_hub_token <huggingface-token> \
--max_steps 1_000 \
--per_device_train_batch_size 32
```

- On a multi-node distributed environment, we can use Slurm in the head node to run a job on multiple compute nodes. 

  - First, please prepare some scripts in the `/lustre` directory. See `scripts/job.slurm`, and `scripts/train.sh` for reference. The `job.slurm` file sets up some Slurm environment variables, and uses `srun` to execute the `train.sh` script, which uses `torchrun` (see reference [here](https://pytorch.org/docs/stable/elastic/run.html)) to execute the above-mentioned `run_mlm.py` script with settings related to the distributed environment.

  - From `/lustre`, submit the job by running `sbatch job.slurm`. A job ID will be returned to console output.

  - Can check the log stored at `slurm_<job-id>.out`.

- Since the main purpose of this repo is to show how to use ParallelCluster to perform pre-training for a LLM, we only train the model for 1000 steps here, which takes less than 10 minutes. Feel free to adjust the parameters for a full training.

- If you are interested in how to use a pre-trained LLM model for inference with or without fine-tuning, feel free to take a look at this repo: https://github.com/delongmeng/Machine-Translation-LLM-Finetuning.


### Troubleshooting

- If you get "Insufficient Capacity Errors" (ICEs) when creating the cluster, you may consider EC2 [capacity reservation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-reservations-using.html).

- The output of the OnNodeConfigured script for head node can be found here: `/var/log/cfn-init-cmd.log` and `/var/log/cfn-init.log`.

- The output of the OnNodeConfigured script for compute node can be found here: `/var/log/cloud-init-output.log`.


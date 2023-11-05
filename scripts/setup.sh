#!/bin/bash

echo "[POST-INSTALL] "
# Configure Linux
. /etc/os-release
OS=$NAME

. /etc/parallelcluster/cfnconfig
HOME_DIR=${1:-/home/$cfn_cluster_user}
EBS_SHARED_DIR=$cfn_ebs_shared_dirs

sudo apt --fix-broken install -y 
sudo apt install git-lfs

echo "[POST-INSTALL] the node type is $cfn_node_type"

# set up NCCL and EFA in compute nodes
if [[ $cfn_node_type == "ComputeFleet" ]]; then

	# start configuration of NCCL and EFA only if CUDA and EFA present
	CUDA_DIRECTORY=/usr/local/cuda
	EFA_DIRECTORY=/opt/amazon/efa
	OPENMPI_DIRECTORY=/opt/amazon/openmpi
	if [ -d "$CUDA_DIRECTORY" ] && [ -d "$EFA_DIRECTORY" ]; then

	    # installing NCCL
	    NCCL_DIRECTORY=$HOME_DIR/nccl
	    if [ ! -d "$NCCL_DIRECTORY" ]; then
	        echo "[POST-INSTALL] Installing NVIDIA nccl"
	        cd $HOME_DIR
	        git clone https://github.com/NVIDIA/nccl.git
	        cd $NCCL_DIRECTORY
	        make -j src.build
	    fi

	    # installing aws-ofi-nccl
	    echo "[POST-INSTALL] Installing aws-ofi-nccl"
	    AWS_OFI_DIRECTORY=$HOME_DIR/aws-ofi-nccl

	    if [ ! -d "$AWS_OFI_DIRECTORY" ]; then
	        echo "[POST-INSTALL] Downloading aws-ofi-nccl"
	        cd $HOME_DIR
	        git clone https://github.com/aws/aws-ofi-nccl.git -b aws
	    fi

	    cd $AWS_OFI_DIRECTORY
	    ./autogen.sh
	    ./configure --with-mpi=$OPENMPI_DIRECTORY --with-libfabric=$EFA_DIRECTORY --with-nccl=$NCCL_DIRECTORY/build --with-cuda=$CUDA_DIRECTORY
	    export PATH=$OPENMPI_DIRECTORY/bin:$PATH
	    make
	    sudo make install
	    echo "[POST-INSTALL] Finished installing aws-ofi-nccl"
	fi

fi 

PYTHON_VERSION=$(python --version 2>&1)
echo "[POST-INSTALL] Python version: $PYTHON_VERSION" 

sudo apt --fix-broken install -y 
# Install Python venv and activate Python virtual environment
echo "[POST-INSTALL] installing python3.8-venv..."
sudo apt install python3.8-venv -y

if [[ $cfn_node_type == "HeadNode" ]]; then
  cd $HOME_DIR
  echo "[POST-INSTALL] Activate Python virtual environment"
  python3.8 -m venv ml_venv
  source ml_venv/bin/activate

  PYTHON_VERSION=$(python --version 2>&1)
  echo "[POST-INSTALL] Python version: $PYTHON_VERSION" 

  echo "[POST-INSTALL] installing pip and Python packages"
  pip install -U pip

  # Install packages from repos
	pip install transformers datasets apache-beam jupyter tensorflow flax accelerate
	python -m ipykernel install --user --name ml-venv --display-name "ml-venv"
	# Install Pytorch of a version that supports CUDA 12.1
	pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

  chown ubuntu:ubuntu -R ml_venv 2>&1

else
  DNS_SERVER=""
  grep Ubuntu /etc/issue &>/dev/null && DNS_SERVER=$(resolvectl dns | awk '{print $4}' | sort -r | head -1) 2>&1
  IP="$(host $HOSTNAME $DNS_SERVER | tail -1 | awk '{print $4}')" 2>&1
  DOMAIN=$(jq .cluster.dns_domain /etc/chef/dna.json | tr -d \") 2>&1
  sudo sed -i "/$HOSTNAME/d" /etc/hosts 2>&1
  sudo bash -c "echo '$IP $HOSTNAME.${DOMAIN::-1} $HOSTNAME' >> /etc/hosts" 2>&1
fi


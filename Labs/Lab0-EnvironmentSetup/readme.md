# Setup instructions
## Provision Lab Virtual Machine in Azure
1. VM Type: NC6  
2. VM OS: Ubuntu 16.04 LTS
3. Networking: Default configuration
4. Set DNS name label to your prefered name
5. Open Jupyter port (TCP/8888) in the VM's Network Security Group configuration
## Install CUDA 8.0
1. Connect to your VM
```
ssh <username>@<machine name>
```
2. Install *gcc*
```
sudo apt-get update
sudo apt-get install gcc
```
3. Download and verify CUDA
```
cd
mkdir Downloads
cd Downloads
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
md5sum cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
```
4. Install CUDA
```
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
sudo apt-get update
sudo apt-get install cuda-drivers
```  
6. Verify installation
```
nvidia-smi
```
You should see the output similar to the following

![alt text](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/media/n-series-driver-setup/smi.png)

### Install Anaconda
1. Download Anaconda installer
```
cd ~/Downloads
wget https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
```
2. Install Anaconda 
   
   Use defaults for all prompts but the last one - say *yes* to modifying PATH
```
bash ~/Downloads/Anaconda3-4.4.0-Linux-x86_64.sh
```

   Logout and login again
   
3. Verify installation
```
python
```

### Install CNTK







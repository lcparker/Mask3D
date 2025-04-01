set -e
apt install libopenblas-dev --yes

export TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"

# install cuda 11.3 toolkit
if [ ! -f "cuda_11.3.0_465.19.01_linux.run" ]; then
    wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
fi
apt --purge remove "*cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" --yes --allow-change-held-packages || echo "No CUDA packages found or removal failed"
apt autoremove -y
sudo apt update -y
sudo apt install g++-9 gcc-9 --yes
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
sudo update-alternatives --config gcc


conda create -n mask3d_cuda113 python=3.10 pip=23.3 -y
eval "$(conda shell.bash hook)"
conda activate mask3d_cuda113

conda install -c conda-forge pyyaml==5.4.1 -y
conda install -c conda-forge pycocotools==2.0.4 -y
python -m pip install omegaconf==2.0.6
sh cuda_11.3.0_465.19.01_linux.run --toolkit --silent --override

conda env update -f environment.yml

python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
if [ ! -f $TORCH_SCATTER_WHEEL ]; then
    python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
else
    python -m pip install $TORCH_SCATTER_WHEEL
fi
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps

mkdir -p third_party
cd third_party

if [ -d "MinkowskiEngine" ]; then
    echo "MinkowskiEngine directory already exists."
    
    # Check if it's a git repository
    if [ -d "MinkowskiEngine/.git" ]; then
        echo "Updating existing repository to specific commit..."
        cd MinkowskiEngine
        # Fetch all changes but don't merge them
        git fetch --all
        # Hard reset to the specific commit
        git reset --hard 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
        # Update submodules if any
        git submodule update --init --recursive
    else
        echo "Directory exists but is not a git repository. Removing and cloning fresh..."
        rm -rf MinkowskiEngine
        git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
        cd MinkowskiEngine
        git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
    fi
else
    echo "Cloning MinkowskiEngine repository..."
    git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
    cd MinkowskiEngine
    git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
fi

# Common steps after having the right repository state
rm -f requirements.txt
echo "numpy" > requirements.txt
python setup.py install --force_cuda --blas=openblas

cd ..
if [ ! -d "ScanNet" ]; then
    git clone https://github.com/ScanNet/ScanNet.git
fi
cd ScanNet/Segmentator
git fetch --all
git reset --hard 3e5726500896748521a6ceb81271b0f5b2c0e7d2
make

cd ../../pointnet2
python setup.py install

cd ../../
python -m pip install pytorch-lightning==1.7.2 --no-deps
python -m pip install torchmetrics==1.5.2 --no-deps
pip install -r lightning-1.7.2-requirements.txt

echo "Make sure to export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda-11.3/targets/x86_64-linux/lib"

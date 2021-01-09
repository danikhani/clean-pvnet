# Using PVNet to convert the NDDS exported Data to Preprocessed Linemode.


## Introduction

Thanks [Haotong Lin](https://github.com/haotongl) for providing the clean version of PVNet and reproducing the results. And also [Katharina](https://github.com/KatharinaSchmidt) for writing the main code.

In this repo, we only want to convert the generated datasets.


## Installation

There are two ways to use this repo. 
First way and recomended one: Using the Docker file
second way: Installing directly on the system.

### 1) Using docker:
First off: Im totally new to docker and I used the docker file from [Floris](https://github.com/zju3dv/clean-pvnet/tree/master/docker) and configured it to Katharinas project. I would be gratefull if you share your thoughs about how to make it better.
1. For this you need [Nvidia Docker](https://github.com/NVIDIA/nvidia-docker) installed on your system 
2. clone this repo
3. Use documentation [How to install](docker/README.md) to build and run the docker container.
4. After running the docker you need to install the requirements.txt in the running container.
	
 ```
    pip install -r requirements.txt
 ```
after stoping the container with docker stop #containerID and rerunning it from step 3, the step 4 should be also repeated.
5. Compile cuda extensions under `lib/csrc`:


    ```
    ROOT=/path/to/clean-pvnet
    cd $ROOT/lib/csrc
    export CUDA_HOME="/usr/local/cuda-10.2"
    cd dcn_v2
    python setup.py build_ext --inplace
    cd ../ransac_voting
    python setup.py build_ext --inplace
    cd ../nn
    python setup.py build_ext --inplace
    cd ../fps
    python setup.py build_ext --inplace
    ```

### 2) Installing directly on the system:
1. Set up the python environment:
    ```
    conda create -n pvnet python=3.7
    conda activate pvnet

    # install torch 1.1 built from cuda 10.2
    pip install torch==1.1.0 torchvision==0.2.1

    pip install Cython==0.28.2
    sudo apt-get install libglfw3-dev libglfw3
    pip install -r requirements.txt
    ```
2. Compile cuda extensions under `lib/csrc`:
    ```
    ROOT=/path/to/clean-pvnet
    cd $ROOT/lib/csrc
    export CUDA_HOME="/usr/local/cuda-10.2"
    cd dcn_v2
    python setup.py build_ext --inplace
    cd ../ransac_voting
    python setup.py build_ext --inplace
    cd ../nn
    python setup.py build_ext --inplace
    cd ../fps
    python setup.py build_ext --inplace

    # If you want to use the uncertainty-driven PnP
    cd ../uncertainty_pnp
    sudo apt-get install libgoogle-glog-dev
    sudo apt-get install libsuitesparse-dev
    sudo apt-get install libatlas-base-dev
    python setup.py build_ext --inplace
    ```
## Using the code

Add your .ply model to the folder /data. Rename your model to "model.ply" and run the give3d script with "python give3d.py". model.yml with the linemod format of 8 corners of the cubiod will be generated in folder data.


## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{peng2019pvnet,
  title={PVNet: Pixel-wise Voting Network for 6DoF Pose Estimation},
  author={Peng, Sida and Liu, Yuan and Huang, Qixing and Zhou, Xiaowei and Bao, Hujun},
  booktitle={CVPR},
  year={2019}
}
```

## Acknowledgement

This work is affliated with ZJU-SenseTime Joint Lab of 3D Vision, and its intellectual property belongs to SenseTime Group Ltd.

```
Copyright SenseTime. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

This repository has been forked and modified to work with data from [NDDS](https://github.com/NVIDIA/Dataset_Synthesizer).

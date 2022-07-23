# DSCollection

_A collection of tools ease manipulations of datasets._

## Installation 

Choose one of the two types of DSCollection packages to install:
- `python3 -m pip install dscollection`
  - This is the standard version of *DSCollecton* which contains all tasks except **generate**.
  - The standard version has fewer dependencies thus is a perfect entry for quick start.
- `python3 -m pip install dscollection[gst]`
  - This is the full-fledged version od *DSCollection* contains all tasks.
  - You need correctly install **Gst-Python** (python bingding of Gstreamer) and **pyds** (Python binding of DeepStream) 
    in order to use **generate** task.
  - **To install pyds, you need at least one GPU card.**

### 1. Installation dependencies
DSCollection heavily relies on `opencv-python`. Since users may have their own opencv installed, therefore after installing 
the package, you may need install opencv yourself if it does not exist.

For users who installed `dscollection[gst]`, go over the sections below.

**Prerequisite**:
- At least 1 NVIDIA GPU card;
- CUDA ToolKit, CuDNN and TensorRT have already installed.

> **TIP:**
>If you have trouble to install the prerequisites, I wrote a [blog](https://blog.csdn.net/qq_27370437/article/details/124945605?spm=1001.2014.3001.5501) that may save your time.

#### Install gst-python and pyds

- Ubuntu 20.04 LTS
  - Refer to deepstream [official site](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#remove-all-previous-deepstream-installations) 
  and deepstream python github [link](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/blob/master/HOWTO.md) 

- Ubuntu 22.04 LTS
  - Since the version of Gstreamer in Ubuntu 22.04 LTS is **1.20.1** by default, you need install the corresponding gst-python at first.
  - Download [gst-python-1.20.1](https://gstreamer.freedesktop.org/src/gst-python/gst-python-1.20.1.tar.xz)
  - Install gst-python-1.20.1
    - ```shell
      # install meson and ninja 
      sudo apt install meson ninja-build
      
      # build gst-python
      cd ~/Downloads/gst-python-1.20.1
      meson builddir && cd builddir
      ninja
      
      # copy gst-python dynamic library to gi/overrides
      GI_OVERRIDES_PATH=$($(which python) -c 'import gi; import os; print(os.path.join(os.path.dirname(gi.__file__), "overrides"))')
      cp ./gi/overrides/* $GI_OVERRIDES_PATH
      ```
- Install [deepstream sdk site](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#remove-all-previous-deepstream-installations)
- Install libyaml-cpp0.6
  - ```shell
    # uninstall libyaml-cpp0.7
    sudo apt purge libyaml-cpp-dev
      
    # Install libyaml-cpp0.6
    cd ~/Downloads
    git clone https://github.com/jbeder/yaml-cpp.git
    cd yaml-cpp
    git checkout tags/yaml-cpp-0.6.3
      
    mkdir build && cd build
    cmake -DYAML_BUILD_SHARED_LIBS=ON ..
    make
    sudo make install
      
    # create a soft link under /usr/lib
    sudo ln -s /usr/local/lib/libyaml-cpp.so.0.6.3 /usr/lib/libyaml-cpp.so.0.6
    ```
- Install [deepstream-python-bingdings](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/blob/master/bindings/README.md), go to section 1.3 directly:
  - ```shell
    # download deepstream-python
    cd ~/Downloads
    git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git
    cd deepstream_python_apps
    git checkout tags/v1.1.3
    git submodule update --init
    
    # add the new certificates that gst-python git server now uses
    sudo apt-get install -y apt-transport-https ca-certificates -y
    sudo update-ca-certificates
    
    # IMPORTANT: Since the built-in python3 version in Ubuntu 22.04 is 3.10, 
    # but deepstream only supports python3.8, therefore you must create a 
    # virtual environment with python=3.8
    conda create -n deepstream python=3.8
    conda activate deepstream
    
    # IMPORTANT: Edit CMakeList.txt to let pybind11 find <Python.h>
    # Add a new line `~/miniconda3/envs/deepstream/include/python3.8` at include_directories
    # Probably at line 71!
    
    # IMPORTANT: Edit CMakeList.txt to let pyds to link python3.8 library
    # Add a new line `target_link_directories(pyds PRIVATE ~/miniconda3/envs/deepstream/lib)`
    # at line 88 (right beofore target_link_libraries(pyds pthread ...) )
    
    # Install deepstream bindings
    cd bindings
    mkdir build && cd build
    cmake -DPYTHON_MINOR_VERSION=8 ..
    make -j$(nproc)
    
    # Now you have successfully built pyds-1.1.3*.whl
    # Install pyds, make sure you have activated correctly virtual environment
    pip install pyds-1.1.3-py3-none-linux_x86_64.whl
    ```
    

### 2. Install DSCollection





> NOTE: DSCollection already uploaded to [TestPyPI](https://test.pypi.org/project/dscollection/).


### Task description:

#### 1. **extract**:
Extract some or all classes from dataset, generate a new dataset.

    
The task requires one or more input datasets, each can be of any type, like VOC | KITTI | COCO, etc. 
After extraction process complete, final output could be either VOC or KITTI.

> **NOTE**:
>- Only images and labels are being considered and therefore any
    other files or directories keep unchanged.
>- One can use -c/--contiguous options to force the images and labels
    are renamed sequentially. 
>- Even thought this task can do similar work as `combine`, but is slower
  than latter since this task will load all labels into ImageLabels which
  followed by dataset-type conversions. If the type input and output dataset
  keep unchanged, use `combine` instead.

##### Typical use cases:
```shell
# Use as datatype converter
>> dsc extract -i <path> -o <path> --kitti

# Extract specific classes
>> dsc extract -i <path> -o <path> --kitti --classes person;face

# Extract from multiple inputs with difference types, generate a new VOC
>> dsc extract -c -i <path1> <path2> <path3> -o <path> --voc --classes person;face
```
    
#### 2. **generate:**
Generate a new dataset from raw fisheye videos. This task is built on top of
Gstreamer and Deepstream. One can only use this task if dependencies are all fit.

This task requires one or multiple input videos or directories containing videos,
do some fisheye image preprocessing and perform model inference (if specify models)
on all frames (or fixed skipped frames), then save all processed images and inferenced
labels as a new VOC | KITTI dataset.

#### Typical use cases:

```shell
# Automatic labeling
>> dsc generate -c -i <path1> <path2> -o <path> --kitti --ext .png -m <model1> -m <model2>

# Fisheye image preprocessing
>> dsc generate -c -i <path1> <path2> -o <path> --voc --ext .jpg --crop-size 1100 --cam-shifts -5,0,0,0 --roi-offset 100 --grayscale

```

    
#### 3. **visualize**:
Visualize a dataset using OpenCV or Matplotlib, save annotated images also supported.

The input dataset could be any known dataset type, refer `core/dataset.py` for details.

    
##### Typical use cases:
```shell
# Visualize a dataset (dataset type will be automatically detected)
>> dsc visualize -i <path> --backend 0

# Visualize a dataset, save annotated images
>> dst visualize -i <path> -i <path> --backend 1 --rows 2 --cols 2
```
    
#### 4. **split**:
Split a dataset into multiple parts. Only images and labels are considered, therefore
any other files or directories keep unchanged. The task used shutils.move() function
by default, thus all images and labels from original dataset will be moved to new
destination path.

There are three options to split a dataset:
- Specify number partitions: `-ns/--num-splits`
- Specify number images per each partition: `-ne/--num-per-each`
- Specify percentage of each part: `-nf/--num-fractions`

> **NOTE:** 
If not output path is given, all partitions will saved in the input directory.
Directory name of partitions are prefixed by the name or input dataset, e.g.
VOC2007-part00, VOC2007-part01, ...
unless explicitly given by option --names.

##### Typical use cases:
```shell
# Split a dataset into N parts
>> dsc split -i <path> -ns N [-o <path>]

# Train test split
>> dst split -i <path> -nf 0.85;0.15 --names train;test -o <path>
```
  
#### 5. **combine**:
Combine multiple datasets into a whole new dataset. Only images and labels are
considered, therefore any other files or directories keep unchanged. The task
used shutils.move() function by default, thus all images and labels from original
dataset will be moved to new destination path.

> **NOTE:**
The input path could be `wildcard`.

##### Typical use cases:
```shell
# Combine multiple partitions
>> dsc combine -i <path/VOC2007-part*> --contiguous
```

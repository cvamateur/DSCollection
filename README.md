# DSCollection

_A collection of tools ease manipulations of datasets._

## Installation 

### 1. Build binary wheel and install locally
```shell
python3 setup.py bdist_wheel
pip3 install -e .
# pip install dscollection
```

### 2. Upload to TestPyPI
```shell
pip3 install --upgrade twine

# generate distribution archives 
python3 -m build

# upload distributions to testpypi
twine upload --repository testpypi dist/*
```

### 3. Install newly uploaded package using pip
```shell
pip install -i https://test.pypi.org/simple/ dscollection 
```

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

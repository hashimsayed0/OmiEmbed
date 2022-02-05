# SubOmiEmbed

**SubOmiEmbed: Self-supervised Representation Learning of Multi-omics Data for Cancer Type Classification**

This codebase is built upon https://github.com/zhangxiaoyu11/OmiEmbed

## Introduction

SubOmiEmbed is an extension of OmiEmbed that supports the SSL technique of feature subsetting for the following tasks.

1.  Multi-omics integration
2.  Dimensionality reduction
3.  Omics embedding learning
4.  Tumour type classification
5.  Phenotypic feature reconstruction
6.  Survival prediction
7.  Multi-task learning for aforementioned tasks

## Getting Started

### Prerequisites
-   CPU or NVIDIA GPU + CUDA CuDNN
-   [Python](https://www.python.org/downloads) 3.6+
-   Python Package Manager
    -   [Anaconda](https://docs.anaconda.com/anaconda/install) 3 (recommended)
    -   or [pip](https://pip.pypa.io/en/stable/installing/) 21.0+
-   Python Packages
    -   [PyTorch](https://pytorch.org/get-started/locally) 1.2+
    -   TensorBoard 1.10+
    -   Tables 3.6+
    -   scikit-survival 0.6+
    -   prefetch-generator 1.0+
-   [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 2.7+

### Installation
-   Clone the repo
```bash
git clone https://github.com/hashimsayed0/OmiEmbed
cd OmiEmbed
```
-   Install the dependencies
    -   For conda users  
    ```bash
    conda env create -f environment.yml
    conda activate omiembed
    ```
    -   For pip users
    ```bash
    pip install -r requirements.txt
    ```

### Try it out
-   Train and test using the built-in sample dataset with the default settings
```bash
python train_test.py
```
-   Check the output files
```bash
cd checkpoints/test/
```
-   Visualise the metrics and losses
```bash
tensorboard --logdir=tb_log --bind_all
```


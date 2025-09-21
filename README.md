
<div align="center">

# Modeling Bottom-Up Information Quality During Language Processing

[![Paper EMNLP 2025](http://img.shields.io/badge/paper-arxiv.2206.08672-B31B1B.svg)](https://arxiv.org/...)

</div>


## Installing dependencies

1. Clone the repository

```bash
git clone https://github.com/DiLi-Lab/Bottom-Up-Information.git
cd Bottom-Up-Information/
```

2. Create an environment, e.g with [anaconda](https://www.anaconda.com/products/individual).
```
conda create --name bottomup python=3.10
conda activate bottomup
```
3. Install the necessary packages with pip:
```
pip install -r requirements.txt
```
Alternatively, you can directly create the conda environment and install dependencies using the provided `environment.yml` file:
```
conda env create -f environment.yml -n bottomup
conda activate bottomup
```

## Accessing the Data

### Human Reading Data with MoTR
We use the mouse-tracking for reading (MoTR) paradigm to collect human reading data (see this [paper](https://www.sciencedirect.com/science/article/pii/S0749596X24000378) for details). We provide the post-processed reading measures data in the `data/Human` folder, which was used in the experiments. The codes for post-processing the raw MoTR data are in the `post_processing` folder.

The full range of reading data can be downloaded from [here](https://osf.io/udztq/files/osfstorage), including the raw MoTR data, the mouse association data (anologous to fixations in eye-tracking), and the reading measures data.

The link to the experiment in Chinese is [here](https://cuierd.github.io/Re-Veil/multilingual_motr/zh/).
The link to the experiment in English is [here](https://cuierd.github.io/Re-Veil/multilingual_motr/en/).

### Half-occluded Image Data for MI estimations with LMs
We create noised input with the half-occluded image  to estimate the mutual information (MI) between degraded bottom-up visual information and linguistic representations. We provide the post-processed MI / IG data in the `data/LLM` folder.

The images can be downloaded from [here](https://osf.io/udztq/files/osfstorage), including the full images, the half-occluded images, and some statistics of the images.

The code for generating the half-occluded images is adapted from [here](https://github.com/Belval/TextRecognitionDataGenerator). Note, we fixed some bugs and modified the code to generate half-occluded images. We can provide the modified code upon request, but it is not the main focus of this repository.

## Running the Experiments
The scripts for training the models are in the `src` folder. In addition, we provide scripts we used to check the half-masking effects to ensure the masking is even for upper and lower halves of the images. These scripts start with "check" (e.g., `check_half_character_zh.html`).

In the folder `src/transocr` are codes for training the transocr models (see this [paper](https://dl.acm.org/doi/10.1145/3581783.3611755)). The model architecture is adapted from [this repository](https://github.com/FudanVI/FudanOCR/tree/main/text-focused-Transformers). Though we provide the modified code, to run the code, please follow the instructions in the original repository. 


The bash scripts for running the experiments are in the `scripts` folder. You need to modify the paths and settings in the bash scripts before running them.

The scripts for calling the models and calculating the MI / IG on testing data sets are in the `notebooks` folder. 
The .ipynb files starts with "cal" (e.g., `cal_entropy_en_words.ipynb`) are simple baseline models. 
The .ipynb files starts with "finetune" are for calculating the MI / IG with LMs. 

## Data Analysis and Visualization
The R scripts for data analysis and visualization are in the `analysis` folder.

In the `analysis` folder, the folder `precomputed` and `stats` contain the pre-computed results and statistics used in the analysis. The folder `visualization` contains the generared figures in the paper. 

## Contact
Feel free to create an issue or email Cui Ding (<cui.ding@uzh.ch>) for any questions or suggestions.





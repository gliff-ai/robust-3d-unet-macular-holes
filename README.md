# Robust 3D U-Net Segmentation of Macular Holes
Implementation of this [paper](https://arxiv.org/abs/2103.01299) which allows for the segmentation of macular hole data.

## Setup
To install dependencies:

    pip install -r requirements.txt

## Dataset
We cannot make public the dataset used for the paper due to privacy concerns.
The dataloader expects three folders:
 - train
 - validation
 - test

each with a folder `im` and `gt` within them, corresponding to the OCT image and ground truth image respectively.
All images and ground truths are of the following dimensions: 321x376x49

For convenience, we provide a script to generate synthetic data, to demonstrate this layout and file format:

    python3 generate_macular_holes.py

## Running
To train the models for the paper:


    cd bin
    ./run_train.sh

Perf metrics on train, validation and test sets as it is trained will be in CSV files in `out/cli-seg-results`.

## Inference
To run inference on the trained models for the paper:


    cd bin
    ./run_inference.sh

The output 3D TIFF images will be in `out/cli-seg-infer`.

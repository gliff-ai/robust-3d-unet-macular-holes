# Robust 3D U-Net Segmentation of Macular Holes
[](./img/macular-hole.png)
This repository contains the official code the paper [Robust 3D U-Net Segmentation of Macular Holes](https://arxiv.org/abs/2103.01299) which allows for the segmentation of macular hole data using 3D U-Nets.

Code was written by Jonathan Frawley <a itemprop="sameAs" content="https://orcid.org/0000-0002-9437-7399" href="https://orcid.org/0000-0002-9437-7399" target="orcid.widget" rel="me noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon">https://orcid.org/0000-0002-9437-7399</a>.

If you use this software, please cite it as below:


    @misc{frawley2021robust,
          title={Robust 3D U-Net Segmentation of Macular Holes}, 
          author={Jonathan Frawley and Chris G. Willcocks and Maged Habib and Caspar Geenen and David H. Steel and Boguslaw Obara},
          year={2021},
          eprint={2103.01299},
          archivePrefix={arXiv},
          primaryClass={eess.IV}
    }


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

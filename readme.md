# Semantic Error Prediction: Estimating Word Production

This repository provides the necessary code for creating a semantic error prediction dataset.

The paper introducing this task has been published as part of the 2024 NLP4CALL workshop.

Please cite this paper as:

FORTHCOMING



The majority of code was written by David Strohmaier, with some files provided by Chris Bryant and modified by David Strohmaier, which is indicated by the top line.-


A further repository which allows the training of the models will be made available at a later time.

## Creating the Dataset

We provide scripts for the creation of the data. Follow these steps on bash:
1. Create a conda environment from the provided YAML file:
> conda env create -f requirements.yml
2. Activate the conda environment.
> conda activate bea_extraction
3. Run the script create_files.sh

The data will be in the "data" folder in the "train" and "dev" directories. The "dev" directory contains the data used for evaluation. The new dev split (described in the paper) is in the "train" directory as "new_dev.tsv".

We also provide an additional script (featurise.sh) to provide the feature-based data for the regression reported in the paper.

If you have trouble accessing the data or running the code feel free to contact me at <david.strohmaier@cl.cam.ac.uk>.


## Data

The source of the errors and corrections is the 2019 BEA shared task for Grammatical Error correction.

Due to licensing restrictions, we do not provide the data files themselves directly. They are, however, publicly available at the following URLs:
- https://www.cl.cam.ac.uk/research/nl/bea2019st/#data BEA shared task data
- https://osf.io/kz2px/ AoA data used for regression model

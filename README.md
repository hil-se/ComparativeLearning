This repository contains the code for the comparative learning framework described in "Efficient Story Point Estimation With Comparative Learning".


## Dependencies
The required dependencies must be installed to run the source code.
```
pip install -r requirements.txt
```

## Data
Story point estimation data and its pre-split training, validation and testing splits can be found under Data/GPT2SP/Split/
Simply running PairwiseExperiments.py under Code/ will generate the encodings or word embeddings, and save the pairwise data for each split for each project under Data/GPT2SP/Embeddings/
To generate GPT2 encodings, set the modelType variable in PairwiseExperiments.py to the value "GPT2SP".
To generate FastText word embeddings, set the modelType variable in PairwiseExperiments.py to the value "FTSVM".

## Comparative learning experiments
Once the dependencies are installed, run the corresponding files to run the comparative learning experiments with the default parameters. The parameters values can be changed to conduct different experiments.
The experiments can be run using the command -
```
python PairwiseExperiments.py
```

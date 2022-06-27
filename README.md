# Accelerating Diversity Sampling for Deep Active Learning By Low-Dimensional Representations
Instead of using the high-dimensional latent features of a neural network, we propose to use output vectors for active diversity sampling. 
This code repository contains a small proof-of-concept experiment evaluated on MNIST. 
We compare the following diversity sampling methods:
- `KMeansSampling`: In each acquisition round, perform kmeans clustering and select NN to centroid for annotation.  
- `KCenterGreedy`: In each acquisition round, iteratively select instance with maximum distance to their nearest already labeled instance.  
- `Kmeans++`: In each acquisition round, iteratively select instance with probability proportional to distance of already selected instances in the current batch.    

The diversity concepts are popular choices for various active learning strategies.
We execute the three methods on latent features, pca reduced features and on the probability vectors each.
Our results show that using the output probabilities greatly reduces acquisition times and at the same time increases label efficiency.
 

This repository is built on [Kuan-Hao Huang's DeepAL repository](https://github.com/ej0cl6/deep-active-learning) and [Jordan Ash's BADGE repository](https://github.com/JordanAsh/badge) 

## Install Requirements
```
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Start MLFlow for Tracking
```
(venv) mlflow server --tracking_uri=${TRACKING_URI}
```
We assume that an mlflow instance is running at ```TRACKING_URI```.

## Run Experiments
Start the experiment using the script ```executables/run_v.py```:

```
(venv) PYTHONPATH=. python executables/run_v.py --tracking_uri=${TRACKING_URI}
```

## Visualization
Please run the jupyter notebook "evaluate.ipynb" in folder `notebooks` to visualize our results.   


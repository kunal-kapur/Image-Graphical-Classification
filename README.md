## Image classification using graphical representations

This contains instructions for how the run the given code to reproduce results.
To see results from this project see the report.

### Instructions

Getting started 

First create a virtual environment using python3.10:

```console
python -m venv .venv
source ./.venv/bin/activate
```
For windows do 
```console
source ./.venv/scripts/activate
```
Install packages (only ones for this one are torch and matplotlib)
```console
pip install -r requirements.txt
```

## Dataset 
Download the dataset from 
### https://www.kaggle.com/datasets/borhanitrash/animal-image-classification-dataset
Note the given default folder when calling train.py is Animals, however, this can be changed. 



## Running 
```console
python train.py [args]
```
The list of arguments can be found below 

## Argument list:

| Argument            | Short Form | Default Value | Description                                                  |
|---------------------|------------|---------------|--------------------------------------------------------------|
| `--distance`        | None       | `15`          | The maximum distance for Harris keypoints                    |
| `--epochs`          | `-e`       | `10`          | The number of training epochs.                               |
| `--neighbors`       | `-k`       | `3`           | The number of nearest neighbors to consider.                 |
| `--lr`              | `-l`       | `0.001`       | The learning rate for optimization.                          |
| `--schedule`        | `-s`       | `6`           | The step at which to adjust the learning rate schedule.      |
| `--classified`      | `-c`       | `False`       | Boolean flag to toggle binary classification mode.           |
| `--nodes`           | `-n`       | `20`          | The number of nodes to include in the graph or network.      |
| `--data`            | None       | `Animals`     | The number of nodes to include in the graph or network.      |



#### Binary classification option
The weights of the pretrained binary classifier are included here, however, if you would like to train them run 
```console
python train_binary.py
```
and the results should be in the path train_binary/model_weights folder 


See script.sh to get examples of how to run the script or to run several examples. 


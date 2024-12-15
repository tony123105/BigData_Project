# LEAF: a LEarnable Audio Frontend

LEAF is a learnable alternative to audio features such as mel-filterbanks, that can be initialized as an approximation of mel-filterbanks, and then be trained for the task at hand, while using a very small
number of parameters.

A complete description of the system is available in [our recent ICLR publication](https://openreview.net/forum?id=jM76BCb6F9m).


## docker build

docker build -t {name} .

or 

docker pull ken20020209/bigdata_project:latest

##


git clone https://github.com/1tnatsonC/BigData_Project.git



## Installation

```bash
cd BigData_Project/paper-leaf-audio

pip3 install .
```


## Training audio classification models and get test result

We also provide a basic training library that allows combining a frontend with
a main classification architecture (including PANN), and training it on a classification dataset.

This library uses Gin: `common.gin` contains the common hyperparameters such as
the batch size or the classification architecture. Each frontend then has its own
`.gin` config file that uses all hyperparameters from `common.gin` and overrides
the frontend class. In `leaf_custom.gin` we show how Gin allows to easily change
hyperparameters of the frontend, as well as the main classification architecture
and using SpecAugment.

To train a model on mel-filterbanks:

```bash
python3 -m example.main --gin_config=example/configs/mel.gin
```

or on LEAF:

```bash
python3 -m example.main --gin_config=example/configs/leaf.gin
```


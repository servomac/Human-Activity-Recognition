# Human Activity Recognition from smartphone signals

## Usage

Download the [dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones):

```
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
unzip UCI\ HAR\ Dataset.zip
```

Build the image and run:

```
docker build -t keras .
docker run -it -v `pwd`:/app keras python3 main.py
```

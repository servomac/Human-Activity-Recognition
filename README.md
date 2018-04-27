# Human Activity Recognition from smartphone signals

[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.me/servomac)

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

## Results

Confusion matrix:

```
Pred                LAYING  SITTING  STANDING  WALKING  WALKING_DOWNSTAIRS  WALKING_UPSTAIRS
True
LAYING                 510        0        27        0                   0                 0
SITTING                  0      384       105        1                   0                 1
STANDING                 0       89       443        0                   0                 0
WALKING                  0        0         0      449                  28                19
WALKING_DOWNSTAIRS       0        0         0        0                 419                 1
WALKING_UPSTAIRS         0        5         1        2                  58               405
```

## Resources:

Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. [Human Activity Recognition on Smartphones using a Multiclass Hardware-Friendly Support Vector Machine](https://www.icephd.org/sites/default/files/IWAAL2012.pdf). International Workshop of Ambient Assisted Living (IWAAL 2012). Vitoria-Gasteiz, Spain. Dec 2012 

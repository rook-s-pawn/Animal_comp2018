# Animal_comp2018

Name：rook-s-pawn
## Overview
Predicting gender of bird from its GPS trajectory

## Description
Python scripts to classify GPS trajectories of birds (Calonectris leucomelas) into male or female.
The scripts are originally made for Animal Behavior Challenge: ABC2018.
https://competitions.codalab.org/competitions/16283

Features include "total travel distance", "maximum speed value", "average and median of total travel distance per day", "average, median, and variance of latitude and longitude", and " Average, median, and variance of the total of speeds faster than 1 m / s " were used.
I used it as a feature because I thought that a difference could be seen in both males and females.

## Requirement
Python 3
The following directories are needed to generate in advance.
1. feature_csv
2. fig
3. predict_file
4. result_label

The directory names are described in the python scripts. You can change the directory names.

## Usage
1. feature_make.py

This program creates the features for classification from GPS trajectories.

```$ python feature_make.py```

As a result, the "train_feature.csv" is created.

2. anly.py

The program makes a model to classify the data into male or female with training data.
Before running this program, you need to create "train_feature.csv" in advance.

```$ python anly.py```

As a result, the "model_rf.pkl" is created.

3. predict_test_label.py 

This is a program to classify the test data into male or female using "model_rf.pkl".
Before running this program, you need to create "test_feature.csv" using "feature_make.csv".

```$ python predict_test_label.py```

As a result, the "y_submission.txt" is created.

## Install
```$ pip install vincenty ```

## Contribution
1. CodaLab - Competition (https://competitions.codalab.org/competitions/16283)

## Author
rook-s-pawn

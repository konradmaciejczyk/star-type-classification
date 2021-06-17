# Star type classification

This is a classifier comparison for problem of star types classification.</br>
You can learn more about dataset I used in this project: https://www.kaggle.com/deepu1109/star-dataset

## About dataset
This dataset consist of 240 examples of stars that can be classified as:</br>
Brown Dwarf, Red Dwarf, White Dwarf, Main Sequence, Supergiant or Hypergiant.

Each star is described by:  
Temperature (K), Luminosity(L/Lo), Radius(R/Ro), Absolute magnitude(Mv), Star type, Star color, Spectral Class.</br>
Star Color and Spectral Class are categorical features.

## Classifiers
* My implementation of KNN
* My implementation of NearestCentroid
* Neural Network

## Running code
First of all make sure you have all needed python modules installed on your host.<br/>
The code requires: numpy, matplotlib, pandas, sklearn and tensorflow.

Run:
```
    pip3 install -r requirements.txt
```
to install all needed modules.

* main.py - this is a driver file that consists of all code required for loading, handling data and running tests;
* algorithms.py - here I've written my implementations of some clasifiers I'm using in this project


# Naval-Mine-Identifier-using-ANN

A Deep Neural Network is designed to classify between rocks and mines .
This technology is frequently used in naval mine detection by many countries.

## Modules Used 
- Tensorflow == 1.5
- scikit-learn
- pandas
- matplotlib

## Steps for use

- First create a virtual environment and install tensorflow (ver. 1.5)
- Allow system site packages such as pandas,scikit-learn,numpy,scipy,matplotlib in the virtual environment.
- This repository contains both the training model and the restored model.If you wish to train the model again, execute ***ANN.py***
```
python3 ANN.py
```
else you can use the already trained model by executing ***restore_ANN.py***
```
python3 restore_ANN.py
```
## About the Dataset

Please [Click Here](http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)) to know more about the dataset.

## Images

##### Cost VS Epoch
![picture](https://github.com/jitensinha98/Naval-Mine-Identifier-using-ANN/blob/master/Images/figure_2.png)

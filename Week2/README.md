## Week2 Notes

Implementation of perceptron algorithms are pretty straightforward. 

For evaluating learning classifier important thing to remember is **cross-validation strategy** used when working on a limited data set, which is usually the case in practise. 
The general idea is to split data into k batches and then run evaluation for the classifier k times. In each run testing data is one data sample while everything else is used 
for training. This way, each sample is used 1 time for the testing set and k-1 times in the training set. 

The perceptron algorithms and evaluation algorithms are included in main.py folder. 

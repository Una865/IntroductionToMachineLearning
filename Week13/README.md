## Non-parametric methods

Those methods refer to the ones that do not have fixed parametrization in advance. *It is not the case that you do not have parameters at all!*
1. Trees
2. Nearest Neighbor

Overview:

Non-parametric machine learning algorithms do not make strong assumptions about the form of the mapping function, but still maintaining ability to generalize to unseen data. They learn functional form from the training data. 

Advantages:
- flexibility
- power
- performance 
- more human-interpretable 

Limitations:

- requires more data
- slower to train
- more risk to overfit the model

### Tree models:
partitioning the input space and making different, but simple, predictions on different regions of the space
### Additive models: 
training different classifiers on the whole space and averaging the answers (estimations error is decreased this way)

## Trees
Finding partition of input space, and then fitting simple models in each piece. The partition is done by decision trees, which recursively splits the space. 
Decision trees are learning simple decision rules infered from data features. 

Disadvantages of decision trees:
- they can be unstable as little variations in the data can result in a completely different tree being generated
- If some class dominates, the tree can be biased so it is a good practise to balance tha data before fitting the model
- predictions of decision trees are constant approximations, therefore they are not good in extrapolation 

Example of decision trees:

![alt_txt](https://github.com/Una865/IntroductionToMachineLearning/blob/main/Week13/DecisionTree.png)

Using tree models is most appropriate where the individual input features are **meaningful measurements**. For example, using measurements done on some patient. Tree models are easily interpretable, which is import in medicine for example, for the doctors to understand decision that the algorithm proposed.

### In the dtree.py is implementation of decision tree from scratch.

The way the algorithm works:
- at every node iterate over all features and iterate over all unique values that feature can take 
- use the current value as a separator for the question and split the data
- choose the split with the most information gain
- if no split is made, make it a leaf node (otherwise it is called decision node)

For calculating information gain we need impurity functions. There are few inmupirty function, but they all need to satisfy 3 conditions:
1. achieve maximum at the unifrom distribution
2. achieve zero when one of the probabilities equals to 1
3. to be symetric with regard to their permutations

Possible impurity functions:
1. entropy function 

![alt_text](https://github.com/Una865/IntroductionToMachineLearning/blob/main/Week13/Screenshot%202022-04-11%20at%2023.02.09.png)

2. gini index

![alt_text](https://github.com/Una865/IntroductionToMachineLearning/blob/main/Week13/Screenshot%202022-04-11%20at%2023.05.13.png)

Information gain is then calculated as:

![alt_text](https://github.com/Una865/IntroductionToMachineLearning/blob/main/Week13/Screenshot%202022-04-11%20at%2023.11.41.png)
where **Eparent** refers to the impurity of parent node and **Echild** is the average impurity of child nodes.
Great video: https://www.youtube.com/watch?v=LDRbO9a6XPU



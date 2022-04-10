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
Finding partition of input space, and then fitting simpe models in each piece. The partition is done by decision trees, which recursively splits the space. 
Decision trees are learning simple decision rules infered from data features. 

## Neuron
The basic element of neural netwotk. Its paramaters are weights, bias value and activation function. 

## Activation functions
The reason for including activation function is to include non-linearity in our network. If there is no function computed on neurons, the output of the newtork is actually the linear combination of all the previous neurons, which we do not really want.

There are many activation functions:
- **sigmoid**: range(0,1) , drawbacks: slowly converges, vanishing gradient, output non-zero centered
- **hyperbolic tangent**: range (-1,1), it is never used in output layer, drawbacks: vanishing gradient
- **softmax function**: for multiclass classification problem, never used in hidden layers
- **rectified linear unit ReLu**: 0 - for negative values, ReLu(x) = x otherwise, does not have vanishing problem, drawbacks: dying ReLu problem, as values can be very big it can be computationaly expensive
- **step function**:  values 0 or 1, not usually used today in hidden layers
As an activation function has a large impact on the performance of the network, the question is how to choose one. 

**Additional notes** : Sigmoid generally works better in case of classifiers. 
ReLu should only be used in hidden layers and hyperblic tangent is never used in output layer. There is also leaky ReLU function which also does not have vanishing problem. The difference is that when the value is smaller than 0, this function ouptus value of x multiplied by some small scalar.

## Gradient Descent

There are 3 versions of gradient descent:
- batch gradient descent
- stochastic gradient descent
- mini-batch gradient descent

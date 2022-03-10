## Neuron
The basic element of neural netwotk. Its paramaters are weights, bias value and activation function. 

![alt_text](https://github.com/Una865/IntroductionToMachineLearning/blob/main/Week6/neuron.png)

## Activation functions
The reason for including activation function is to include non-linearity in our network. If there is no function computed on neurons, the output of the newtork is actually the linear combination of all the previous neurons, which we do not really want.

There are many activation functions:
- **sigmoid**: range(0,1) , drawbacks: slowly converges, vanishing gradient, output non-zero centered
- **hyperbolic tangent**: range (-1,1), it is never used in output layer, drawbacks: vanishing gradient
- **softmax function**: for multiclass classification problem, never used in hidden layers
- **rectified linear unit ReLU**: 0 - for negative values, ReLu(x) = x otherwise, does not have vanishing problem, drawbacks: dying ReLu problem, as values can be very big it can be computationaly expensive
- **step function**:  values 0 or 1, not usually used today in hidden layers
As an activation function has a large impact on the performance of the network, the question is how to choose one. 

**Additional notes** : Sigmoid generally works better in case of classifiers. 
ReLu should only be used in hidden layers and hyperblic tangent is never used in output layer. There is also leaky ReLU function which also does not have vanishing problem. The difference is that when the value is smaller than 0, this function ouptus value of x multiplied by some small scalar.

## Gradient Descent
For training neural networks we can use gradient descent algorithm. 

There are 3 versions of gradient descent:
- batch gradient descent
- stochastic gradient descent
- mini-batch gradient descent: compromise between batch and stochastic gradient

## Error backpropagation
My medium post: https://medium.com/@unajacimovic/understanding-how-backpropagation-works-3ed6c6c96f2e

## Initializing weights
Good general purpose strategy is to choose each weight from a Gaussian normal distribution with mean 0 and standard deviation 1/m where m is the number of inputs. Many activation have zero slope when values are very big so we want to keep first values of weights small.

## Optimizing neural network parameters 
- momentum gradient descent very useful link: https://distill.pub/2017/momentum/
- addelta algorithm
- adam algorithm

## Regularization
- early stopping: basic idea is to have samples of data for validation on whoch you compute the loss function after every epoch and if it starts to systematicaly increase, stop your algorithm ad return weights with the lowest cost function
- weight decay: penalizing norm of weights
- adding noise: before each gradient is computed add a small zero-mean normally distributed noise 
- dropout: for deep neural newtorks, prohibiting some units in one forward pass so the network won't rely on this small set of units. This is done by multiplying every activation layer with vector of zeros and ones (not to change activation layers).
- batch normalization: 



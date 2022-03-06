## Linear logistic classifier

Important topics:
- sigmoid function
- loss function
- log loss/cross entropy
- regularization
- Gradient Descent 
- Stochastic Gradient Descent
- Support vector machine

Codes:
- Homework code: here are the codes from the homewrok section, some of it are mine and some are theirs as they provided it or their version was shorter so I decided to use it
- Fish project code: There are two versions of this project. In fish.py I used the codes from the homework to build the SupportVectorMachine model from scratch, while in fish2.py I used scikit-learn (documentation: https://scikit-learn.org/stable/modules/svm.html).


### Sigmoid funcion
New hypothesis class was introduced: linear logistic classifier. Insted of output values only being {-1,+1} with logistic function we now have output ranging from 0 to 1 (zero and 1 are not included because it is not possible to get exactly 1). This is done by using sigmoid function or logistic function.


![alt text](https://github.com/Una865/IntroductionToMachineLearning/blob/main/Week4/sigmoid.png)

When deciding to which class our data sample belongs to, we use some value to be prediction treshold (for example, it can be 0.5). 

### Loss function and log loss
Furthermore, instead of jus searching through the space for the hypothesis that performs good on the training sample we are now going to search for a hypothesis that minimizes the overall loss. Or, in other words, we want to pick parameter values such that maximizes probability assigned by te hypothesis class to the correct output values. This score can be represented as the product of independent probabilities (they provided formulas in the notes: Chapter4 Loss function for logistic classifiers), but the product is hard to deal. Good idea is to work with log of that quantity as the log is monotonic and the log of product is actually sum of logs. That would be the cross-entropy or log loss which we want to maximize. But as I mentioned, we want minimization problem, so we are going to work with the negative value of log loss which we want to minimize.

### Regularization 

Regularization is often introduced for reducing overfitting. Often it is expressed as sum of square values of hypothesis multiplied by some scalar - lambda. By squaring the values and adding to the loss function, we prefer smaller values. Large values are allowed only if they considerably improve the first par of the loss function. Also, lambda is the parameter which we set. When lambda is small we prefer to minimize original loss function and when the labda is big we prefer small hypothesis values. Useful link for more about regularization :
chapter: Improving the way neural networks learn -> Regularization: http://neuralnetworksanddeeplearning.com/chap3.html)

### Gradient Descent 
The goal is to find the minimum of loss function. Why gradient? Well, remember, the gradient tells us how the function's value changes as it's input is changed (NOTE: those changes are very small). So when we are thinking about how to minimize loss function we are actually considering how to change values of our hypothesis so the value of los functon is changes as we want. There is where our gradient helps us. When updating hypothesis values:
1. calculate gradient 
2. go in the negative dierction of the gradient by value of the gradient multiplied with learning rate:
 val = val_prev - learning_rate*gradient(val_prev)
 
 Why the learning rate? It controls how big our steps are. It has to be neither to big nor to low. If the learning rate is too big it may always overshoot and miss the local minimum. If the learning rate is too small, it takes a lot of time for our algorithm to find the local minimum.
 
**Important** : Gradient Descent can not promise to find a global minimum, it can find local minimum which does not have to be a global one. If function is non-convex, were gradient descent converges depends on the initial values of hypothesis.

### Stochastic Gradient Descent 

It is like lazy version of gradient descent algorithm. Insetad of calculating gradients of all data points, averaging it and taking that one big(ish) step, in Stochastic Gradient you chose one data point randomly, compute the graidient as if there were only that point and taking one small step.

### Support Vector Machine

It is a linear classifier which tries to maximize the margin (distance between hyperplane and closest data points of each class).

![alt_text](https://github.com/Una865/IntroductionToMachineLearning/blob/main/Week4/SVM.jpeg)

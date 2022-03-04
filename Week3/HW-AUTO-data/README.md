## AUTO data

For this part of the homework students are asked to use their code file and perform implemented perceptron/averaged_perceptron algorithms and evaluate their
accuracies while varying different parameters and encoding strategies. It is not hard to do so as they already provided every function you need. As a total beginner,
I believe there are others like me that want to understand more deeply what is happening in their code when processing the data as it may seem not pretty
straightforward.

There are 6 columns used for the auto data:
- cylinders 
- displacement
- horsepower
- weight
- acceleration
- origin

Car name was left out in this exercise.
For encoding strategies in this homework it is used:
- cylinders: one hot encoding (cylinders take values from set {3,4,5,6,8} there is no order structure here so it is a good idea to use this strategy to process the data)
- displacement: using standardization
- horsepower: use standardization
- weight: use standardization
- acceleration: use standardization
- origin: use one hot encoding, again origin are values from set {1,2,3} so it is a good strategy to one hot encode it


In order to prepare data features as a list of tuples needs to be in a format:
**('name_of_feature', encoding_strategy(raw/standard/one_hot))**
which will be a parameter to a function **auto_data_and_labels**, which will return transformed data in a desired way.

The **auto_data_and_labels** function will return auto_data in shape (12,392). It may now be confusing why the dimension of our data sample is now 12 as we have 6 columns,
but for he cylinders there is a vector of length 5 ({3,4,5,6,8} are 4 possible values it can take) and the origin has a dimension 3 ({1,2,3} are possible values it can
take. This forms : 5 (cylinders) + 1 (displacement) + 1 (horsepower) + 1 (weight) + 1 (acceleration) + 3 (origin) = 12


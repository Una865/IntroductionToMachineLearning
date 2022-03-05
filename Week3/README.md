# Defining features for data

Throughout excercises in this week I worked on feature extraction. 

When dealing with features it is important to be aware of what kind of data we are dealing with. When looking at numbers, the data is usually divided into discrete and continious, but what is their difference?

## Discrete Data
Discrete values can be reffered to as "the number of". They take specific countable values. Examples of discrete data are:
- number of people in the classroom 
- number of push-ups done
- number of presents

### Encoding strategies for discrete features
Important strategy for procesing data in this homework:
- one hot encoding: when there is no ordering structure it is good to use a vector of length d(number of discrete values it can take) where all values are zero except for the one specific value which is one. One example of this strategy is dealing with colours. If there are 5 colours that occur, you will have vectors of length 5. The 1 will indicate the colour of that data sample, while all other values in the vector would be 0.
- standarization: resaling data to have a mean 0 and a standard deviation of 1, useful link: https://sebastianraschka.com/Articles/2014_about_feature_scaling.html


## Continous Data
It is any measured value within a specific range. Examples of continious data are:
- the temperature of a room
- the car speed
- the height of children

NOTE: FeatureEngineeringCar and fashionMNIST are my experimenting and the rest are implementations of homework.

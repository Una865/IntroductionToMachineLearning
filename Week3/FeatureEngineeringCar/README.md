# Defining features for data

When dealing with features it is important to be aware of what kind of data we are dealing with. When looking at numbers, the data is usually divided into discrete and continious, but what is their difference?

## Discrete Data
Discrete values can be reffered to as "the number of". They take specific countable values. Examples of discrete data are:
- number of people in the classroom 
- number of push-ups done
- number of presents

### Encoding strategies for discrete features
Strategies covered in course are:
- one hot encoding: when there is no ordering structure it is good to use a vector of length d(number of discrete values it can take) where all values are zero except for the one specific value which is one. One example of this strategy is dealing with colours. If there are 5 colours that occur, you will have vectors of length 5. The 1 will indicate the colour of that data sample, while all other values in the vector would be 0.
- thermometer code: if values have a natural ordering, but they are not naturaly mapped into real numbers, a good idea is to use vector of lenght d (d as the greatest value) where first k values (0<=k<=d) are 1 and the rest are 0.

## Continous Data
It is any measured value within a specific range. Examples of continious data are:
- the temperature of a room
- the car speed
- the height of children

## MNIST dataset

In this part the goal was to make linear classifier which will classify pictures to be one or another digit. Images are loaded in format (28,28).

Four ways of feature extraction were used here:
- raw 0-1 features: (28,28) reshape to (784,1) and use in perceptron algorithm
- row-average: (m,n) or in this case (28,28), transform to (m,1), in this case (28,1), where element i is the average value of previous row i isnpired by the fact that some numbers take more hoirzontal space
- column-average: (m,n), in this case (28,28), transform to (n,1), in this case (28,1), where element i is the average of the previous column
- top-botom_average: (m,n), in this case (28,28), transform to (2,1) where the first element represent the average value of the upper half of the image, while the second represents average value of the bottom half of the image

For each of the four ways there is a function which implements the feature extraction. 

I computed three combinations:
- 0 vs 1
- 3 vs 6
- 8 vs 9

Evaluations of accuracies of each run are in the files results0vs1.txt, results3vs6.txt and results8vs9.txt.

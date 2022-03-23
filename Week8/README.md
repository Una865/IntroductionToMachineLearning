## Convolutional Neural networks-CNNs

-  filters
-  max pooling 
-  weight sharing

Filters are used for spatial detection of patterns. For example, when classifying object on an image we need to be able to detect it no matter where it is on th picture, which is quite hard with vanilla neural networks. In one of my scripts is an example of a simple filter that detcts left edges.
One good example:

![alt_txt](https://github.com/Una865/IntroductionToMachineLearning/blob/main/Week8/Screenshot%202022-03-23%20at%2020.43.29.png)

These networks are very useful in computer vision. For example, when detecting the object in the picture, we need to detect it correctly no matter where it is on the picture. This is quite hard with vanilla neural networks. 


Additional:
Autoencoders: my medium post - https://medium.com/mlearning-ai/autoencoders-a-scary-name-d961bbc9dcea

Scripts:
**images.py:** very simple object counter. Uses a filter that detects all left-edges in 1d array
**CNNforText.py**: using convolutional neural network on movie reviews

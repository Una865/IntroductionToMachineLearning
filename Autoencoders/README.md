Autoencoders

The goal of this project is to find optimal image compression that could later be used for object classification. Size of images are 32x32x3 and we want to be able to have a coding part as 1-D vector from which tne network can reconstruct the image back.

Different architectures can be found in:

1. ConvolutionalAE1.py 
2. ConvolutionalAE2.py 
3. ConvolutionalAE3.py :
For this part I tried 4 different sizes of the code layer, and the results are shown:

100x1

![alt_text](https://github.com/Una865/IntroductionToMachineLearning/blob/main/Autoencoders/Screenshot%202022-03-26%20at%2009.05.50.png)


300x1

![alt_text](https://github.com/Una865/IntroductionToMachineLearning/blob/main/Autoencoders/CNNae3%20-%20reconstructed.png)


600x1

![alt_text](https://github.com/Una865/IntroductionToMachineLearning/blob/main/Autoencoders/CNNae3%20600x1.png)


1000x1

![alt_text](https://github.com/Una865/IntroductionToMachineLearning/blob/main/Autoencoders/CNNae3%201000x1.png)


2000x1

![alt_text](https://github.com/Una865/IntroductionToMachineLearning/blob/main/Autoencoders/ConvolutionalAE3%20-%202000x1.png)

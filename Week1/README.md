## Week1 Notes

The machine learning algorithm is the algorithm that is able to learn from the data. Machine learning can help us tackle the problems that are otherwise too difficult to solve with fixed set of rules or proograms designed by humans. So, the main focus of machine learning is making decisions or predictions based on data.

Machine learning tasks are usually desxribes as hos the machine should process an example. 
Most common machine learning tasks:

1. *Classification:* specifying which of k categories  input belongs to
2. *Classification with missing inputs*: doing classification even if some of the input measurements are missing. The machine learning algorithm now needs to learn a set of functions that corresoponds to classifying input (x) with a different subset of its input missing. To efficiently define such a large set of functions is to learn a probability distribution over all of the relevant variables.However, we only need to learn the joint probability distribution
3. *Regression*: here the computer needs to output numerical value given some input. This is similar to classifcation, but here we have different format
4. *Transcription*: here the computer needs to learn how to get discrete, textual form from the relatively unstructured representation of some kind of data. One example is extracting sentences from image or producing words out of audio 
5. *Machine translation*:convert sequence of input into another sequence (commonly applied to languages)
6. *Structures output*: any task where output is the vector with important relationships between the different elements (Example: image segmentation). 
7. *Anomaly detetction:* flag some object as normal/anomal (Example: fraud detection. If someone steals your card their purchaces will probably be from different distrubution than yours)
8. *Synthesis and sampling*: generating new examples that are similar to those in the training data. For example, geneating textures of large objects or spoken version of sentences.
9. *Imputation of missing values*: predicting missing values of x
10. *Denosiing*: Input is corrupted image from example and the goal is to get the original image
11. *Denisty estimation or probability mass function*: the algorithm needs to learn the structure of the probabilitu distribution

Kind of machine learning algorithms:
1. *Unsupervised learning:* given a dataset we want to find patterns or structure inherent in it. In other words, we want to learn the probability distribution that generated the dataset
2. *Supervised learning*: each example of dataset has its label or target  
3. *Reinforcement learning*: here we do not experience the fixed dataset. There is no training set a priori. The problem is framed as agent interacting with the environment 

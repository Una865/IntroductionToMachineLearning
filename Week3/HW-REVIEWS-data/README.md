## Bag of words

In this excercise the aim is to get to know more about bag of words.

The basic idea is to make dictionary of all the words that occur in the document and make feature vectors of lenght which is the length of the dictionary with 1 at the index
which is the index of that word in that data sample in the dictionary.

One small example:  if my file has two data samples with followint sentences
- Today the weather was good and I was happy.
- She was unhappy because she did not feel good.

The dictionary would be :

**{'Today':0, 'She':1, 'the':2,'weather':3,'unhappy':4, 'was':5,'good':6,'because':7,'and':8,'I':9,'happy':10,'did':11,'not':12,'feel':13}**

The feature vectors for these two sentences would then be:
- [1,0,1,1,0,1,1,0,1,1,1,0,0,0]
- [0,1,0,0,1,1,0,1,0,0,0,1,1,1]

The feature vectors do not have to strictly have 1 or 0. Sometimes we can use feature vectors with numbers indicating frequency of the word in data sample. 

The main feature of Bag-of-Words model is that it does not perserve order.



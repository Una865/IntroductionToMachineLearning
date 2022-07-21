## Sequential models

- states
- markov decision processes
- policy of behaviour

State machine is a description of process in terms of its potential sequnces of data.

Markov decision process is a variation on a state machine in which the transition function is a probabilistic distribution over the next state given the previous state and input. The output is the state and some states are more desirable. 

**Markov property** : new state and reward depend only on the previous state and action that is taken, not on the whole history of states and actions.

**Markov process** : is defined by the probabilities of transition function 

The policy function is a function which assing probabilistic distribution among the set of actions to the state S. The role of the rewards is to implicitly define the goal of the agent. They should be set as to when the agent wants to maximize the reward, in the same tame he is doing the desired work. The often mistake is to set rewards for agent on how to do something, instead of what to do. 

**Recurrent neural newtorks**

RNNs are constructed with an idea to model dependencies between instances. In those tasks it is usually unknown how many of previous instances are influencing the next element. Also, for different elements it may be the case that the number of previous instances that are influencing the current one is different. Elements of the sequence which is the input, are processed one by one, sequentially. 

Some of the variations of RNNS:
 - one to one
 - many to many
 - one to many
 - many to one
 
 RNNs have problems with vanishing gradients and keeping long dependencies. Hiddent state is computed as linear combination of previous state and input, thus effect of previous states is fast vanished by the influence of new information. The architecture which is used as a solution are LSTMs.
 
 The basic idea for LSTMs is to the cell which keeps ifnormation about hidden state with the control of reading, writing and forgetting. Therefore we have:
 - input gate
 - forget gate
 - output gate
 
 The formulas are:
 
 ![alt_txt](https://github.com/Una865/IntroductionToMachineLearning/blob/main/Week9/Screenshot%202022-07-21%20at%2018.18.23.png)
 
 Every gate has same structure as a regular RNN cell and with the give input and its parameters decides at which level the operation it controls will take place. 
 
 ![alt_txt](https://github.com/Una865/IntroductionToMachineLearning/blob/main/Week9/1*z7bEceNfH6X_N75HA9kyoA.png)
 
 

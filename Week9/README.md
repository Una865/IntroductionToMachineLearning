## Sequential models

- states
- markov decision processes
- policy of behaviour

State machine is a description of process in terms of its potential sequnces of data.

Markov decision process is a variation on a state machine in which the transition function is a probabilistic distribution over the next state given the previous state and input. The output is the state and some states are more desirable. 

**Markov property** : new state and reward depend only on the previous state and action that is taken, not on the whole history of states and actions.

**Markov process** : is defined by the probabilities of transition function 

The policy function is a function which assing probabilistic distribution among the set of actions to the state S. The role of the rewards is to implicitly define the goal of the agent. They should be set as to when the agent wants to maximize the reward, in the same tame he is doing the desired work. The often mistake is to set rewards for agent on how to do something, instead of what to do. 

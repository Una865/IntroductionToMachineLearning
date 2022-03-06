## Regression

Important topics:
- discrete values -> real-valued
- structual error
- estimation error

The difference now is that values in the linear classification were {+1,-1} and now with regression we want to predict real-values.
First step in building this model is choosing loss function. In this homework mean squared error is used, which also have easy computational gradient.

Not just in regression, but machine learning in general, we should be aware of two ways hypothesis may contribute to error:
- structural error: when we can not represent a good hypothesis 
- estimation error: this error occurs when we do not have enough data

When experimenting with lambda we should also be aware that increasing its value would decrease estimation error but increase structural error and vice versa.

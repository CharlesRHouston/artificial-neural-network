# Artificial Neural Network (From Scratch)

### Description

Python class which allows the user to easily construct and train a feed-forward neural network. Coded up entirely from scratch without the use of Tensorflow or PyTorch. The class is contained in `neural_network.py` and experiments are performed in `main.py`, including comparisons of optimisers, learning rate decay, and weight initialisation. 

### Documentation

ArtificialNeuralNetwork attributes:

* **hidden**         - list of neurons (as integers) in hidden layers; default = [16, 16]
* **problem**        - one of "regression"/"binary_classification"; default = "regression"
* **activation**     - one of "sigmoid"/"tanh"; default = "sigmoid"
* **regularization** - one of "L2"/"dropout"; default = "L2"
* **initializer**    - one of "Xavier"/"He"; default = "Xavier"
* **optimizer**      - one of "Gradient Descent"/"Momentum"/"RMSprop"/"Adam"; default = "Gradient Descent"
* **check_gradient** - boolean e.g. True/False; compares gradient descent to central difference; default = False

ArtificialNeuralNetwork.optimize method:

* **X**              - numpy array of predictor variables (ensure shape has two dimensions, (m, n_x), where m is the number of examples and n_x the dimension of the features) 
* **Y**              - numpy array of response variables (ensure shape has two dimensions, (m, n_y), where m is the number of examples and n_y is the dimension of the target)
* **X_val**          - numpy array of validation set predictor variables (ensure shape has two dimensions, (m, n_x))
* **Y_val**          - numpy array of validation set response variables (ensure shape has two dimensions, (m, n_y))
* **iters**          - integer; default = 1e4
* **learn0**         - learning rate (initial learning rate if decay_rate is not 0); default = 0.01 
* **decay_rate**     - inverse time decay; default = 0
* **lambd**          - L2 weight decay value; default = 0
* **keep_prob**      - dropout value; default = 1.0
* **beta1**          - momentum value; default = 0.9
* **beta2**          - RMSprop value; default = 0.999
* **epsilon**        - for Adam numerical stability; default = 1e-8

Module requirements: numpy, matplotlib

### References

Inspired by: [Deep Learning Specialisation](https://www.coursera.org/specializations/deep-learning) on Coursera
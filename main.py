# CONTENTS
# 1. Documentation
# 2. Simulate data
# 3. Run class
# 4. Regularization
# 5. Optimizer comparison
# 6. Learning rate decay comparison
# 7. Initialisation comparison


#### 1. DOCUMENTATION ####

"""
ArtificialNeuralNetwork attributes:

hidden         - list of neurons (as integers) in hidden layers; default = [16, 16]
problem        - one of "regression"/"binary_classification"; default = "regression"
activation     - one of "sigmoid"/"tanh"; default = "sigmoid"
regularization - one of "L2"/"dropout"; default = "L2"
initializer    - one of "Xavier"/"He"; default = "Xavier"
optimizer      - one of "Gradient Descent"/"Momentum"/"RMSprop"/"Adam"; default = "Gradient Descent"
check_gradient - boolean e.g. True/False; compares gradient descent to central difference; default = False

ArtificialNeuralNetwork.optimize method:

X              - numpy array of predictor variables (ensure shape has two dimensions, (m, n_x))
Y              - numpy array of response variables (ensure shape has two dimensions, (m, n_y))
X_val          - numpy array of validation set predictor variables (ensure shape has two dimensions, (m, n_x))
Y_val          - numpy array of validation set response variables (ensure shape has two dimensions, (m, n_y))
iters          - integer; default = 1e4
learn0         - learning rate (initial learning rate if decay_rate is not 0); default = 0.01 
decay_rate     - inverse time decay; default = 0
lambd          - L2 weight decay value; default = 0
keep_prob      - dropout value; default = 1.0
beta1          - momentum value; default = 0.9
beta2          - RMSprop value; default = 0.999
epsilon        - for Adam numerical stability; default = 1e-8

Module requirements: numpy, matplotlib
"""

from neural_network import ArtificialNeuralNetwork
import numpy as np
import matplotlib.pyplot as plt


#### 2. SIMULATE DATA ####

N = 200

# x

np.random.seed(1)
x = np.random.uniform(-1, 1, N)
x = np.reshape(x, (N, 1))

# y

np.random.seed(2)
y = 2*x + np.sin(2*np.pi*x) + np.random.normal(0, 0.3, (N, 1))
y = np.reshape(y, (N, 1))

# target

XX = np.reshape(np.linspace(-1, 1, 1000), (-1, 1))
YY = 2*XX + np.sin(2*np.pi*XX)

# train and validate split

x_train = x[:100,]
x_val   = x[100:,]
y_train = y[:100,]
y_val   = y[100:,]

# plot data and target function

plt.figure(1)
plt.plot(x_train, y_train, 'ro', label = "Simulated data")
plt.plot(XX, YY, color = "black", label = "Target function")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Simulated data and target function")
plt.legend()


#### 3. RUN CLASS ####

model = ArtificialNeuralNetwork(hidden = [12, 12], activation = "tanh", regularization = "L2", optimizer = "Gradient Descent")
res = model.optimize(x_train, y_train, x_val, y_val, check_gradient = True, lambd = 0.01, decay_rate = 0.002, iters = int(2e4))

# optimize method returns the following entities

opt_parameters      = res[0] # optimized parameters
loss_per_iteration  = res[1] # loss as a function of iteration (epoch)
prediction          = res[2] # predicted values
train_error         = res[3] # training set error
val_error           = res[4] # validation set error

# plot loss per iteration

plt.figure(2)
plt.plot(loss_per_iteration, color = "blue")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Gradient descent: loss per iteration")


#### 4. REGULARIZATION ####

# lambda values to explore
M_seq = 6
lam = np.exp(np.linspace(-7, 0, M_seq))

# error storage
Train_E = np.zeros(M_seq)
Val_E   = np.zeros(M_seq)

# establish model
regular_model = ArtificialNeuralNetwork(optimizer = "Adam", activation = "tanh", initializer = "He")

# convergence plot
plt.figure(3)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Convergence of regularized models")

# optimize for each level of regularization
for i in range(M_seq):
    print("\nModel:", i+1)

    # optimize
    np.random.seed(2021)
    res = regular_model.optimize(x_train, y_train, x_val, y_val, lambd = lam[i], iters = int(1e4), learn0 = 0.01)
    
    loss_per_iteration  = res[1]
    train_error         = res[3]
    val_error           = res[4]
    
    # store train and validation errors
    Train_E[i] = train_error
    Val_E[i]   = val_error
    
    # check convergence on plot
    plt.figure(3)
    plt.plot(loss_per_iteration)

# plot loss vs lambda
plt.figure(4)
plt.plot(lam, Train_E, 'b-o', label = "Train Error")
plt.plot(lam, Val_E, 'r-o', label = "Val Error")
plt.xlabel("$\lambda$")
plt.ylabel("Error")
plt.title("Error as a function of $\lambda$ regularization")
plt.legend()

# optimum lambda

opt_lam = lam[np.argmin(Val_E)]

# train with best lambda

opt_params = regular_model.optimize(x_train, y_train, x_val, y_val, lambd = opt_lam, iters = int(1e4), learn0 = 0.01)[0]

# get predictions

fit = regular_model.neural_net(XX, YY, opt_params)[1]

# compare best model to target function

plt.figure(5)
plt.plot(x_train, y_train, 'ro', label = "Training data", ms = 2)
plt.plot(XX, YY, label = "Target function", color = "black")
plt.plot(XX, fit, label = "Best model", color = "green")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Best model relative to target function")
plt.legend()


#### 5. OPTIMIZER COMPARISON ####

# convergence plot
plt.figure(6)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Comparison of optimizers")

# run through optimizers
optimizers = ["Gradient Descent", "Momentum", "RMSprop", "Adam"]
for optim in optimizers:
    optim_model = ArtificialNeuralNetwork(optimizer = optim)
    np.random.seed(2021)
    res = optim_model.optimize(x_train, y_train, iters = int(1e4), learn0 = 0.001)
    
    loss_per_iteration = res[1]
    
    plt.figure(6)
    plt.plot(loss_per_iteration, label = optim)

plt.figure(6)
plt.legend()


#### 6. LEARNING RATE DECAY COMPARISON ####

learn_model = ArtificialNeuralNetwork()
np.random.seed(1)
res1 = learn_model.optimize(x_train, y_train, learn0 = 1, decay_rate = 0.001)
res2 = learn_model.optimize(x_train, y_train, learn0 = 1)

plt.figure(7)
plt.plot(res1[1][5:], label = "With rate decay", color = "red")
plt.plot(res2[1][5:], label = "Without rate decay", color = "green")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Comparison of rate decay")
plt.legend()

#### 7. INITIALISATION COMPARISON ####

np.random.seed(1)
res1 = ArtificialNeuralNetwork(initializer = "Xavier").optimize(x_train, y_train, iters = 1000, learn0 = 0.0001)
res2 = ArtificialNeuralNetwork(initializer = "He").optimize(x_train, y_train, iters = 1000, learn0 = 0.0001)
res3 = ArtificialNeuralNetwork(initializer = "Mean0Var50").optimize(x_train, y_train, iters = 1000, learn0 = 0.0001)

plt.figure(8)
plt.plot(res1[1], label = "Xavier")
plt.plot(res2[1], label = "He")
plt.plot(res3[1], label = "Mean0Var50")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Comparison of initializers")
plt.legend()


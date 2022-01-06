# MODULES
import numpy as np
import copy

# MODEL CLASS
class ArtificialNeuralNetwork:
    def __init__(self, hidden = [16, 16], problem = "regression", activation = "sigmoid", \
        regularization = "L2", initializer = "Xavier", optimizer = "Gradient Descent"):
        
        self.hidden = hidden
        self.problem = problem
        self.activation = activation
        self.regularization = regularization
        self.initializer = initializer
        self.optimizer = optimizer
    
    # ACTIVATION FUNCTION
    def act1(self, z):
        if self.activation == "sigmoid":
            return 1/(1 + np.exp(-z))
        elif self.activation == "tanh":
            return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
        else:
            raise ValueError("Invalid activation function")
    
    def act1_(self, z): # derivative of act1
        if self.activation == "sigmoid":
            return 1/(1 + np.exp(-z))*(1 - 1/(1 + np.exp(-z)))
        elif self.activation == "tanh":
            return 1 - ((np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z)))**2
        else:
            raise ValueError("Invalid activation function")
    
    # OUTPUT ACTIVATION
    def act2(self, z):
        if self.problem == "regression":
            return z
        elif self.problem == "binary_classification":
            return 1/(1 + np.exp(-z))
        else:
            raise ValueError("Invalid problem type")
    
    def act2_(self, z): # derivative of act2
        if self.problem == "regression":
            return 1 + 0*z
        elif self.problem == "binary_classification":
            return 1/(1 + np.exp(-z))*(1 - 1/(1 + np.exp(-z)))
        else:
            raise ValueError("Invalid problem type")
    
    # LOSS FUNCTION
    def loss(self, yhat, y):
        if self.problem == "regression":
            return 1/2*(yhat - y)**2
        elif self.problem == "binary_classification":
            return -y*np.log(yhat) - (1 - y)*np.log(1 - yhat)
        else:
            raise ValueError("Invalid problem type")
    
    def loss_(self, yhat, y): # derivative of loss
        if self.problem == "regression":
            return yhat - y
        elif self.problem == "binary_classification":
            return (yhat - y)/(yhat*(1 - yhat))
        else:
            raise ValueError("Invalid problem type") 
    
    # NEURAL NET: L2 OR DROPOUT
    def neural_net(self, X, Y, parameters, lambd = 0, keep_prob = 1):
        if self.regularization == "L2":
            # shapes
            m = X.shape[0] # number of examples

            # number of layers
            L = len(parameters) // 2

            # forward propagation
            activations = {}
            combination = {}
            activations["A0"] = X.T 
            for i in range(L-1):
                combination["Z" + str(i+1)] = np.dot(parameters["W" + str(i+1)], activations["A" + str(i)]) + parameters["b" + str(i+1)]
                activations["A" + str(i+1)] = self.act1(combination["Z" + str(i+1)])
            combination["Z" + str(L)] = np.dot(parameters["W" + str(L)], activations["A" + str(L-1)]) + parameters["b" + str(L)]
            activations["A" + str(L)] = self.act2(combination["Z" + str(L)]) 

            # working gradients
            work_grad = {}
            work_grad["d" + str(L)] = self.loss_(activations["A" + str(L)], Y.T)*self.act2_(combination["Z" + str(L)])
            for i in reversed(range(1, L)):
                work_grad["d" + str(i)] = np.dot(parameters["W" + str(i+1)].T, work_grad["d" + str(i+1)])*self.act1_(combination["Z" + str(i)])

            # backward propagation
            gradients = {}
            for i in range(1, L+1): # reversed(range(1, L+1)):
                gradients["dW" + str(i)] = np.dot(work_grad["d" + str(i)], activations["A" + str(i-1)].T)/m + (lambd/m)*parameters["W" + str(i)]
                gradients["db" + str(i)] = np.sum(work_grad["d" + str(i)], axis = 1, keepdims = True)/m 

            # regularized error
            error = np.mean(self.loss(activations["A" + str(L)], Y.T))
            for i in range(L):
                error += (lambd/m)*np.sum(parameters["W" + str(i+1)])

            # prediction 
            prediction = activations["A" + str(L)].T

            return error, prediction, gradients
        
        elif self.regularization == "dropout":            
            # shapes
            m = X.shape[0] # number of examples
        
            # number of layers
            L = len(parameters) // 2
        
            # forward propagation
            A = {} # activations
            Z = {} # linear combinations
            D = {} # dropout array
            A["A0"] = X.T 
            for i in range(1, L):
                Z["Z" + str(i)] = np.dot(parameters["W" + str(i)], A["A" + str(i-1)]) + parameters["b" + str(i)]
                A["A" + str(i)] = self.act1(Z["Z" + str(i)])
            
                ### dropout in forward propagation
                D["A" + str(i)] = np.random.rand(A["A" + str(i)].shape[0], A["A" + str(i)].shape[1])
                D["A" + str(i)] = (D["A" + str(i)] < keep_prob).astype(int)
                A["A" + str(i)] = A["A" + str(i)]*D["A" + str(i)]
                A["A" + str(i)] = A["A" + str(i)]/keep_prob # inverse dropout
            
            Z["Z" + str(L)] = np.dot(parameters["W" + str(L)], A["A" + str(L-1)]) + parameters["b" + str(L)]
            A["A" + str(L)] = self.act2(Z["Z" + str(L)]) 
        
            # working gradients
            gradients = {}
            dZ = {}
            dZ["dZ" + str(L)] = self.loss_(A["A" + str(L)], Y.T)*self.act2_(Z["Z" + str(L)])
            for i in reversed(range(1, L)):
                dA = np.dot(parameters["W" + str(i+1)].T, dZ["dZ" + str(i+1)])
                dA = dA*D["A" + str(i)]
                dA = dA/keep_prob # inverse dropout
                dZ["dZ" + str(i)] = dA*self.act1_(Z["Z" + str(i)])
        
            # backward propagation
            for i in range(1, L+1):
                gradients["dW" + str(i)] = np.dot(dZ["dZ" + str(i)], A["A" + str(i-1)].T)/m
                gradients["db" + str(i)] = np.sum(dZ["dZ" + str(i)], axis = 1, keepdims = True)/m
            
            # regularized error
            error = np.mean(self.loss(A["A" + str(L)], Y.T))
            
            # prediction 
            prediction = A["A" + str(L)].T
            
            return error, prediction, gradients
  
        else:
            raise ValueError("Invalid choice of regularization")
    
    # INITIALIZE WEIGHTS: XAVIER OR HE
    def initialize_weights(self, units):
        if self.initializer == "Xavier":
            parameters = {}
            for i in range(len(units)-1):
                parameters["W" + str(i+1)] = np.random.randn(units[i+1], units[i])*np.sqrt(1/units[i])
                parameters["b" + str(i+1)] = np.ones((units[i+1], 1))
            return parameters
    
        elif self.initializer == "He":
            parameters = {}
            for i in range(len(units)-1):
                parameters["W" + str(i+1)] = np.random.randn(units[i+1], units[i])*np.sqrt(2/units[i])
                parameters["b" + str(i+1)] = np.ones((units[i+1], 1))
            return parameters
        
        elif self.initializer == "Mean0Var50":
            parameters = {}
            for i in range(len(units)-1):
                parameters["W" + str(i+1)] = np.random.normal(0, 50, (units[i+1], units[i]))
                parameters["b" + str(i+1)] = np.ones((units[i+1], 1))
            return parameters
        
        else:
            raise ValueError("Invalid weight initializer")
    
    # INITIALIZE OPTIMIZER
    def initialize_optimizer(self, units):
        if self.optimizer == "Adam":
            V = {} # Momentum
            S = {} # RMSprop
            for i in range(len(units)-1):
                V["dW" + str(i+1)] = np.zeros((units[i+1], units[i]))
                V["db" + str(i+1)] = np.zeros((units[i+1], 1))
                S["dW" + str(i+1)] = np.zeros((units[i+1], units[i]))
                S["db" + str(i+1)] = np.zeros((units[i+1], 1))
            return V, S
        
        elif self.optimizer == "Momentum":
            V = {} # Momentum
            for i in range(len(units)-1):
                V["dW" + str(i+1)] = np.zeros((units[i+1], units[i]))
                V["db" + str(i+1)] = np.zeros((units[i+1], 1))
            return V
        
        elif self.optimizer == "RMSprop":
            S = {} # RMSprop
            for i in range(len(units)-1):
                S["dW" + str(i+1)] = np.zeros((units[i+1], units[i]))
                S["db" + str(i+1)] = np.zeros((units[i+1], 1))
            return S
        
        else:
            raise ValueError("Invalid optimizer")

    # OPTIMIZER: ADAM OR MOMENTUM OR RMSPROP OR GRADIENT DESCENT
    def optimize(self, X, Y, X_val = None, Y_val = None, iters = int(1e4), learn0 = 0.01, \
        decay_rate = 0, lambd = 0, keep_prob = 1, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, \
            check_gradient = False):
        
        if self.optimizer == "Adam":
            # initialise learning rate
            learn = learn0 

            # units
            units = [X.shape[1]] + self.hidden + [Y.shape[1]]
            
            # storage of loss for each iteration
            loss_per_iteration = np.zeros(iters)
        
            # initialise weights and biases and optimizer parameters
            parameters = self.initialize_weights(units)
            V, S = self.initialize_optimizer(units)
        
            # length of parameters
            L = len(parameters) // 2
            
            # corrections
            V_correct = {}
            S_correct = {}
        
            for i in range(iters):
                # forward and back propagation
                error, prediction, grads = self.neural_net(X, Y, parameters, lambd, keep_prob)
                loss_per_iteration[i] = error
                
                # inverse time learning rate decay
                if decay_rate:
                    learn = (1/(1 + decay_rate*iters))*learn0
                
                # update parameters
                for i in range(1, L+1):
                    # momentum
                    V["dW" + str(i)] = beta1*V["dW" + str(i)] + (1 - beta1)*grads["dW" + str(i)]
                    V["db" + str(i)] = beta1*V["db" + str(i)] + (1 - beta1)*grads["db" + str(i)]
                    V_correct["dW" + str(i)] = V["dW" + str(i)]/(1 - beta1**(i))
                    V_correct["db" + str(i)] = V["db" + str(i)]/(1 - beta1**(i))
                    
                    # RMSprop
                    S["dW" + str(i)] = beta2*S["dW" + str(i)] + (1 - beta2)*grads["dW" + str(i)]**2
                    S["db" + str(i)] = beta2*S["db" + str(i)] + (1 - beta2)*grads["db" + str(i)]**2
                    S_correct["dW" + str(i)] = S["dW" + str(i)]/(1 - beta2**(i))
                    S_correct["db" + str(i)] = S["db" + str(i)]/(1 - beta2**(i))
                    
                    # weight updates (Adam)
                    parameters["W" + str(i)] = parameters["W" + str(i)] - learn*V_correct["dW" + str(i)]/(np.sqrt(S_correct["dW" + str(i)]) + epsilon)
                    parameters["b" + str(i)] = parameters["b" + str(i)] - learn*V_correct["db" + str(i)]/(np.sqrt(S_correct["db" + str(i)]) + epsilon)

            # training set error  
            train_error = self.neural_net(X, Y, parameters)[0]
            print("Training set error:", round(train_error, 4))
            
            # validation set error
            val_error = None
            if X_val is not None and Y_val is not None:
                val_error = self.neural_net(X_val, Y_val, parameters)[0]
                print("Validation set error:", round(val_error, 4))
            
            # gradient checking
            if check_gradient:
                print(self.grad_check(X, Y, parameters))

            return parameters, loss_per_iteration, prediction, train_error, val_error
        
        elif self.optimizer == "Momentum":
            # initialise learning rate
            learn = learn0 

            # units
            units = [X.shape[1]] + self.hidden + [Y.shape[1]]

            # storage of loss for each iteration
            loss_per_iteration = np.zeros(iters)
      
            # initialise weights and biases and optimizer parameters
            parameters = self.initialize_weights(units)
            V = self.initialize_optimizer(units)
      
            # length of parameters
            L = len(parameters) // 2
            
            # corrections
            V_correct = {}
      
            for i in range(iters):
                # forward and back propagation
                error, prediction, grads = self.neural_net(X, Y, parameters, lambd, keep_prob)
                loss_per_iteration[i] = error
                
                # inverse time learning rate decay
                if decay_rate:
                    learn = (1/(1 + decay_rate*iters))*learn0
        
                # update parameters
                for i in range(1, L+1):
                    # momentum
                    V["dW" + str(i)] = beta1*V["dW" + str(i)] + (1 - beta1)*grads["dW" + str(i)]
                    V["db" + str(i)] = beta1*V["db" + str(i)] + (1 - beta1)*grads["db" + str(i)]
                    V_correct["dW" + str(i)] = V["dW" + str(i)]/(1 - beta1**(i))
                    V_correct["db" + str(i)] = V["db" + str(i)]/(1 - beta1**(i))
                    
                    # weight updates (Adam)
                    parameters["W" + str(i)] = parameters["W" + str(i)] - learn*V_correct["dW" + str(i)]
                    parameters["b" + str(i)] = parameters["b" + str(i)] - learn*V_correct["db" + str(i)]
            
            # training set error  
            train_error = self.neural_net(X, Y, parameters)[0]
            print("Training set error:", round(train_error, 4))
            
            # validation set error
            val_error = None
            if X_val is not None and Y_val is not None:
                val_error = self.neural_net(X_val, Y_val, parameters)[0]
                print("Validation set error:", round(val_error, 4))
            
            # gradient checking
            if check_gradient:
                print(self.grad_check(X, Y, parameters))
            
            return parameters, loss_per_iteration, prediction, train_error, val_error

        elif self.optimizer == "RMSprop":
            # initialise learning rate
            learn = learn0 
            
            # units
            units = [X.shape[1]] + self.hidden + [Y.shape[1]]

            # storage of loss for each iteration
            loss_per_iteration = np.zeros(iters)
            
            # initialise weights and biases and optimizer parameters
            parameters = self.initialize_weights(units)
            S = self.initialize_optimizer(units)
            
            # length of parameters
            L = len(parameters) // 2
            
            # corrections
            S_correct = {}
      
            for i in range(iters):
                # forward and back propagation
                error, prediction, grads = self.neural_net(X, Y, parameters, lambd, keep_prob)
                loss_per_iteration[i] = error
                
                # inverse time learning rate decay
                if decay_rate:
                    learn = (1/(1 + decay_rate*iters))*learn0
                
                # update parameters
                for i in range(1, L+1):

                    # RMSprop
                    S["dW" + str(i)] = beta2*S["dW" + str(i)] + (1 - beta2)*grads["dW" + str(i)]**2
                    S["db" + str(i)] = beta2*S["db" + str(i)] + (1 - beta2)*grads["db" + str(i)]**2
                    S_correct["dW" + str(i)] = S["dW" + str(i)]/(1 - beta2**(i))
                    S_correct["db" + str(i)] = S["db" + str(i)]/(1 - beta2**(i))
                    
                    # weight updates (Adam)
                    parameters["W" + str(i)] = parameters["W" + str(i)] - learn*grads["dW" + str(i)]/(np.sqrt(S_correct["dW" + str(i)]) + epsilon)
                    parameters["b" + str(i)] = parameters["b" + str(i)] - learn*grads["db" + str(i)]/(np.sqrt(S_correct["db" + str(i)]) + epsilon)
      
            # training set error  
            train_error = self.neural_net(X, Y, parameters)[0]
            print("Training set error:", round(train_error, 4))
            
            # validation set error
            val_error = None
            if X_val is not None and Y_val is not None:
                val_error = self.neural_net(X_val, Y_val, parameters)[0]
                print("Validation set error:", round(val_error, 4))
            
            # gradient checking
            if check_gradient:
                print(self.grad_check(X, Y, parameters))
            
            return parameters, loss_per_iteration, prediction, train_error, val_error

        elif self.optimizer == "Gradient Descent":
            # initialise learning rate
            learn = learn0 
            
            # units
            units = [X.shape[1]] + self.hidden + [Y.shape[1]]

            # storage of loss for each iteration
            loss_per_iteration = np.zeros(iters)
      
            # initialise weights and biases and optimizer parameters
            parameters = self.initialize_weights(units)
            
            # length of parameters
            L = len(parameters) // 2
      
            for i in range(iters):
                # forward and back propagation
                error, prediction, grads = self.neural_net(X, Y, parameters, lambd, keep_prob)
                loss_per_iteration[i] = error
                
                # inverse time learning rate decay
                if decay_rate:
                    learn = (1/(1 + decay_rate*iters))*learn0
                
                # update parameters
                for i in range(1, L+1):
                    # weight updates
                    parameters["W" + str(i)] = parameters["W" + str(i)] - learn*grads["dW" + str(i)]
                    parameters["b" + str(i)] = parameters["b" + str(i)] - learn*grads["db" + str(i)]
            
            # training set error  
            train_error = self.neural_net(X, Y, parameters)[0]
            print("Training set error:", round(train_error, 4))
            
            # validation set error
            val_error = None
            if X_val is not None and Y_val is not None:
                val_error = self.neural_net(X_val, Y_val, parameters)[0]
                print("Validation set error:", round(val_error, 4))
            
            # gradient checking
            if check_gradient:
                print(f"Gradient checking: {self.grad_check(X, Y, parameters)}")
            
            return parameters, loss_per_iteration, prediction, train_error, val_error
    
        else:
            raise ValueError("Invalid optimizer")
            
    # GRADIENT CHECKING FUNCTIONS
    
    def dict2vec(self, dictionary):
        count = 0
        for key in dictionary.keys():
            vector = np.reshape(dictionary[str(key)], (-1, 1))
            if count == 0:
                theta = vector
            else:
                theta = np.concatenate((theta, vector), axis = 0)
            count += 1
        return theta

    def vec2dict(self, vector, parameters):
        dictionary = {}
        start_idx = 0
        for key in parameters.keys():
            shape = parameters[str(key)].shape
            end_idx = start_idx + shape[0]*shape[1]
            dictionary[str(key)] = np.reshape(vector[start_idx:end_idx,0], shape)
            start_idx += shape[0]*shape[1]
        return dictionary
    
    def grad_check(self, X, Y, parameters):
      
      # calculate gradients using backpropagation
      error, prediction, backprop_grads = self.neural_net(X, Y, parameters)
      
      # convert backprop gradients to a vector
      backprop_grads = self.dict2vec(backprop_grads)
      
      # central difference offset
      eps = 0.001
      
      # convert parameters to a vector
      theta = self.dict2vec(parameters)
      
      # storage for central difference gradients
      nparameters = len(theta)
      central_diff = np.zeros((nparameters, 1))
      
      # calculate gradients using central difference
      for i in range(nparameters):
          param_plus = copy.deepcopy(theta)
          param_plus[i] = param_plus[i] + eps
          
          param_minus = copy.deepcopy(theta)
          param_minus[i] = param_minus[i] - eps
          
          error_plus, _, _ = self.neural_net(X, Y, self.vec2dict(param_plus, parameters))
          error_minus, _, _ = self.neural_net(X, Y, self.vec2dict(param_minus, parameters))
          
          central_diff[i] = (error_plus - error_minus)/(2*eps)
      
      # compare the difference using norms - should ideally be ~10^-7
      difference = np.linalg.norm(central_diff - backprop_grads)/(np.linalg.norm(central_diff) + \
                                                                  np.linalg.norm(backprop_grads))
      
      return difference
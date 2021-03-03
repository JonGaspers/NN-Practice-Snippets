import numpy as np

def sigmoid(x): 
    #Activation function f(x) = 1 / (1+e^(-x))
    return 1 / (1 + np.exp(-x))

#setup the class for the neurons
class Neuron:
    #constructor for neuron, store each of the weights & bias
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    #feedforward
    def feedforward(self, inputs):
        # weight inputs, add bias, then use activation function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

weights = np.array([0,1]) # w1 = 0, w2 = 1
bias = 4                 # b = 4
n = Neuron(weights, bias)

x = np.array([2,3])     # x1 = 2, x2 = 3
#print(n.feedforward(x))    #0.999

# neural net class
class NeuralNetwork:
    '''
    Neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)
    Each neuron has the same weights/bias:
    - w = [0,1]
    - b = 0
    '''
    def __init__(self):
        weights = np.array([0,1])
        bias = 0

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        h1_out = self.h1.feedforward(x)
        h2_out = self.h2.feedforward(x)

        #the inputs for o1 are the outputs from h1 & h2
        o1_out = self.o1.feedforward(np.array([h1_out, h2_out]))
        return o1_out

def mse_loss(y_true, y_pred):
    '''
        y_true = answer sheet
        y_pred = NN's predictions
        MSE = mean squared error
        mse = ((y_true - y_pred) ** 2).mean()
    '''
    return ((y_true - y_pred) ** 2).mean()


network = NeuralNetwork()
x = np.array([2,3])
#print(network.feedforward(x))


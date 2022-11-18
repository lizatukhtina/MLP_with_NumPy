import numpy as np

class Linear:

    '''
    in_features: number of features in each sample
    out_features: number of neurons in the next layer
    '''

    def __init__(self, in_features, out_features):

        self.weights = 0.01 * np.random.randn(in_features, out_features)
        self.biases = np.zeros((1, out_features))

    def forward(self, x):

        self.x = x # feature matrix
        # linear transformation
        self.output = np.dot(x, self.weights) + self.biases

    def backward(self, grad):

        # the 'impact' of each weight on the final function
        self.dweights = np.dot(self.x.T, grad) # impact of weights
        self.dbiases = np.sum(grad, axis=0, keepdims=True) # impact of biases
        self.new_grad = np.dot(grad, self.weights.T) # impact of each iput


    def step(self, learning_rate=0.01):

        self.learning_rate = learning_rate
        self.weights += - self.learning_rate * self.dweights # weights update
        self.biases += -self.learning_rate * self.dbiases # biases update


class ReLU:

    def forward(self, x):
        self.x = x
        self.output = np.maximum(0, x)

    def backward(self, prev_grad):

        self.new_grad = prev_grad.copy()
        self.new_grad[self.x <= 0] = 0


class Sigmoid:

    def forward(self, x):

        self.output = 1 / (1 + np.exp(-x))

        return self.output

    def backward(self, grad):

        new_grad = self.output * (1 - self.output) * grad
        self.new_grad = new_grad

    def step(self, learning_step):

        pass


class BCELoss:

    def forward(self, p_hat, y_true):
        y_true = np.expand_dims(y_true, 1)
        samples_losses = -y_true*np.log(p_hat)-(1-y_true)*np.log(1-p_hat) # get losses for all samples
        batch_loss = np.mean(samples_losses) # final mean loss for a batch

        return batch_loss

    def backward(self, pred_grad, y_true):
         batch_size = len(pred_grad)
         # claculate gradient w.r.t sigmoid output
         self.new_grad = ((-y_true.T / pred_grad.T) + (1-y_true.T)/(1-pred_grad.T))
         self.new_grad = (self.new_grad / batch_size).T # normalize gradient


class NeuralNetwork:

    def __init__(self, modules):

        self.modules = modules

    def forward(self, x):

        input = x
        for module in self.modules:
            module.forward(input)
            input = module.output

        probabilities = input # values from the last layer (output)
        return probabilities

    def backward(self, grad):
        '''
        grad: gradient from loss function
        '''
        grad = grad
        
        for module in self.modules[::-1]:
            module.backward(grad)
            grad = module.new_grad

    def step(self, learning_rate):

        # take linear layers
        linear_layers = [m for m in self.modules if isinstance(m, Linear)]
        for linear in linear_layers:
            linear.step(learning_rate) # update weights and biases

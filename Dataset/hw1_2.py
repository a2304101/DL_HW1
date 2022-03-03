# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function

import sys
from sklearn.decomposition import PCA
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from mpl_toolkits import mplot3d
import scipy.io

data = scipy.io.loadmat('spam_data.mat')
ax = plt.axes(projection='3d')

class LinearTransform(object):
	def forward(self, W, b, x):
		self.x = x 
		self.W = W 
		self.b = b 
		return np.dot(self.x, self.W) + self.b
		
	#calculate derivative
	def calc_grad(self, activation, signal):
		w_grad = np.dot(activation.T, signal)
		b_grad = np.sum(signal, axis = 0, keepdims = True)
		return w_grad, b_grad
   #???
	#propagate error signal
	def propagate(self, error_signal, W):
		signal = np.dot(error_signal, W.T)
		return signal  

# This is a class for a ReLU layer max(x,0)
class ReLU(object):
	def forward(self, x):
		return np.maximum(x, 0)
	#???	
	def backward(self, signal, activation):
		signal[activation <= 0] = 0
		return signal

		
class SigmoidCrossEntropy(object):
    def forward(self, input):
        logist =  1 / (1 + np.exp(-input))	
        return logist
    def cost2(self, input, label): 
        logist =  1 / (1 + np.exp(-input))
        cost = -np.sum(np.multiply(label, np.log(logist))+np.multiply((1-label), np.log(1-logist)))/len(input)
        return logist,cost 
		

# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, input_dims, hidden_units, num_labels):
        # INSERT CODE for initializing the network
        self.hidden_units = hidden_units
        self.input_dims = input_dims
        #???
        self.momentum = np.zeros(4)
		
		#random initialization of weights
        self.W1 = np.random.normal(0, 0.5, [input_dims, hidden_units]) 
        self.b1 = np.zeros((1, hidden_units))
        self.W2 = np.random.normal(0, 0.5, [hidden_units, num_labels]) 
        self.b2 = np.zeros((1, num_labels))
		
		#initialization of momentum
        self.W1_m = np.zeros((input_dims, hidden_units))
        self.b1_m = np.zeros((1, hidden_units))
        self.W2_m = np.zeros((hidden_units, num_labels)) 
        self.b2_m = np.zeros((1, num_labels))
		

    def train(self, x_batch, y_batch, learning_rate, gamma):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.forward_pass(x_batch, y_batch)
        self.backward_pass(x_batch, y_batch)
		
		#update weights
        self.W1 -= self.W1_m
        self.b1 -= self.b1_m
        self.W2 -= self.W2_m
        self.b2 -= self.b2_m
		

		
		
    def forward_pass2(self, x_data):
        transformed_inputs = LinearTransform().forward(self.W1, self.b1, x_data) # first linear transformation
        self.relu_activation = ReLU().forward(transformed_inputs) #output of hidden ReLU layer
        transformedDeep_outputs = LinearTransform().forward(self.W2, self.b2, self.relu_activation) #second linear transform
        self.y_hat = SigmoidCrossEntropy().forward(transformedDeep_outputs) #sigmoid output and cost
        self.predictions = np.around(self.y_hat)
        print ( self.predictions )
        return self.predictions
        

    def forward_pass(self, x_data, y_data):
        transformed_inputs = LinearTransform().forward(self.W1, self.b1, x_data) # first linear transformation
        self.relu_activation = ReLU().forward(transformed_inputs) #output of hidden ReLU layer
        transformedDeep_outputs = LinearTransform().forward(self.W2, self.b2, self.relu_activation) #second linear transform
        self.y_hat, self.cost = SigmoidCrossEntropy().cost2(transformedDeep_outputs, y_data) #sigmoid output and cost
        self.predictions = np.around(self.y_hat)
        
       

    def get_minibatches(self, X_train, y_train, minibatch_size): 
        self.minibatch_size = minibatch_size
        minibatches = []
        X_train, y_train = shuffle(X_train, y_train)
        for i in range(0, X_train.shape[0], self.minibatch_size):
            X_train_mini = X_train[i:i + self.minibatch_size]
            y_train_mini = y_train[i:i + self.minibatch_size]
            minibatches.append((X_train_mini, y_train_mini))
            return minibatches
		
	
    def backward_pass(self, x_batch, y_batch):
        error_signal = (self.y_hat - y_batch)/self.minibatch_size
		
        error_signal_hidden = LinearTransform().propagate(error_signal, mlp.W2)
        ReLU_signal_hidden = ReLU().backward(error_signal_hidden, self.relu_activation)
	
        gradient_W2, gradient_b2 = LinearTransform().calc_grad(self.relu_activation, error_signal)
        gradient_W1, gradient_b1 = LinearTransform().calc_grad(x_batch, ReLU_signal_hidden)

		#update momentum
        self.W1_m = self.learning_rate * gradient_W1 + self.W1_m * self.gamma
        self.b1_m = self.learning_rate * gradient_b1 + self.b1_m * self.gamma
        self.W2_m = self.learning_rate * gradient_W2 + self.W2_m * self.gamma
        self.b2_m = self.learning_rate * gradient_b2 + self.b2_m * self.gamma
	
		
    def evaluate(self, x_eval, y_eval):
        self.errors = 0
        self.forward_pass(x_eval, y_eval)
		
        for i, k in enumerate(self.predictions):
            if (k - y_eval[i]).any():
                self.errors += 1
                
        self.accuracy = (1-self.errors/len(y_eval))*100
		
        print('The accuracy is', self.accuracy, '%')
        print('The cross entropy cost is', self.cost)
        print ("Loss: \n" + str(np.mean(np.square(y_eval - self.predictions)))) # mean sum squared loss
        print ("Loss: \n" + str(np.sqrt(np.sum(np.square(y_eval - self.predictions))/len(y_eval)))) # mean sum squared loss
        return self.accuracy		

#Object saver utility function
def save_object(python_object, filename):
	with open(filename+".pickle","wb") as f:
		pickle.dump(python_object,f)


		


if __name__ == '__main__':
	
    #data = pickle.load(open("spam_data.mat","rb"), encoding='bytes')
    train_x = normalize(data['train_x'])
    train_y = data['train_y']
    test_x = normalize(data['test_x'])
    test_y = data['test_y']
    train_x_all = normalize(data['train_x'])
    train_y_all = data['train_y']
    
    pca = PCA(n_components=3)
    X_pca_train = pca.fit_transform(train_x)
    X_pca_test = pca.transform(test_x)
    
     
    
    
    
    num_features, input_dims = train_x.shape
	
    hidden_units = 1000
    learning_rate = 0.0005
    minibatch_size = 32
	
    friction = .75
	
    num_batches = int(num_features/minibatch_size)
	
	
	
    num_labels = 2
    #This is the training regimen each doing 1000 epochs
    minibatch_sizes = [32, 64, 128, 256, 1000, 10000]
    learning_rates = [0.001, 0.0001, 0.00001, 0.00001, 0.00001, 0.00001]
    num_epochs = 100
	
    test_acc = []
    train_acc = []
	
	
    #for friction_ in frictions:
    #for size in hidden_layers:
    #for batch_size in minibatch_sizes:
    #for rate in learning_rates:
    mlp = MLP(input_dims, hidden_units, num_labels)
	
    aass = []
    # with open("final_model.pickle","rb") as f:
        # mlp =  pickle.load(f)
    for i in range(len(minibatch_sizes)):
        for epoch in range(num_epochs):
            minibatches = mlp.get_minibatches(train_x, train_y, minibatch_sizes[i])
            for i, batch in enumerate(minibatches):
            
                mlp.train(batch[0], batch[1], learning_rates[i], friction)
			
            #append epoch accuracy
            print('Train Accuracy')
            train_acc.append(mlp.evaluate(train_x_all, train_y_all))
            print('Test Accuracy')
            test_acc.append(mlp.evaluate(test_x, test_y))
    
         
    predictions=(mlp.forward_pass2(train_x))		
    a2=[]
    for x in range(len(predictions)):
        if predictions[x,0]==1:
            a2.append('r')
        if predictions[x,0]==0:
            a2.append('b')
            
    xdata = X_pca_train[:,0]
    ydata = X_pca_train[:,1]
    zdata = X_pca_train[:,2]
    ax.scatter3D(xdata, ydata, zdata ,color=a2 );     
    #ax.scatter3D(xdata, ydata, zdata ,color=a2 );  
        	
    train_acc = np.array(train_acc).reshape(-1, len(minibatch_sizes)*num_epochs)
    test_acc = np.array(test_acc).reshape(-1, len(minibatch_sizes)*num_epochs)

	
    with open("final_modellarge.pickle","wb") as f:
        pickle.dump(mlp,f)
		
    save_object(train_acc, 'train_final2')
    save_object(test_acc, 'test_final2')

	
    # with open("errors_hiddenlayers.pickle","rb") as f:
        # scores =  pickle.load(f)
	
	
    # # with open("200layer5000iteration.pickle","rb") as f:
        # # mlp =  pickle.load(f)

# coding: utf-8

# ### Rosenblatt's Perceptron Rule
# Frank Rosenblatts perceptron learning rule based on the MCP neuron model <br>
# (F. Rosenblatt, The Perceptron, a Percieving and Recognizing Automaton. Cornell Aeronautical Laboratory, 1957).
# 
# 
# Algorithm that automatically learns the optimal weight coefficients that are then multiplied with the input features in order to make the decision of whether a neuron is executed or not. (Supervised learning & binary classification)<br>
# 
# <b>Initial perceptron rule can be summaried by the following steps<br></b>
# 1. Initalize the weights to 0 or small random numbers.
# 2. For each training sample, $x^i$ perform the following steps:
#     1. Compute the output value $ŷ$.
#     2. Update the weights
# 
# 

# In[24]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%matplotlib notebook
get_ipython().magic('matplotlib inline')

plt.rcParams['figure.figsize'] = (20, 10)

get_ipython().magic('load_ext version_information')
get_ipython().magic('version_information numpy, scipy, matplotlib, pandas')


# In[25]:

class Perceptron(object):
    """
    Perceptron classifier.
    
    Parameters
    ----------
    eta : float, learning rate (between 0.0 and 1.0)
    n_inter : int, number of passes over the training dataset.
    
    
    Attributes
    ----------
    w_ : 1d-array, weights after fitting
    errors_ : list, number of misclassifications in every epoch
    
    """
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):
        """
        Fitting the training data
        
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features], 
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Y : {array-like}, shape = [n_samples], Target values
        
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """ Calculate the net input """
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """ Return class label after unit step """
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# In[26]:

# dataset located here; https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
df = pd.read_csv('data/irisdata-perceptron.csv', header=None)


# In[27]:

df.tail()


# In[28]:

df.info()


# In[29]:

# Extract the first 100 class labels
y = df.iloc[0:100, 4].values

# Convert labels into two integer class labels
# 1 (Versicolor), -1 (Setosa) 

y = np.where(y == 'Iris-setosa', -1, 1)

# Extract the first feature (sepal length) and third feature (petal length)
X = df.iloc[0:100, [0, 2]].values

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length', fontsize='xx-large')
plt.ylabel('petal length', fontsize='xx-large')
plt.legend(loc='upper left', fontsize='xx-large', )


# In[30]:

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')

plt.xlabel('Epochs', fontsize='xx-large')
plt.ylabel('Number of misclassifications', fontsize='xx-large')

plt.show()

# Perceptron converged after the sixth epoch.


# In[31]:

from matplotlib.colors import ListedColormap


# In[32]:

def plot_decision_regions(X, y, classifier, resolution=0.02):
    
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)


# In[33]:

plot_decision_regions(X, y, classifier=ppn)

plt.xlabel('sepal length [cm]', fontsize='xx-large')
plt.ylabel('petal length [cm]', fontsize='xx-large')
plt.legend(loc='upper left', fontsize='xx-large', )


# In[ ]:




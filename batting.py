# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 14:32:15 2017

@author: Josh

mlb batting nn
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# IMPORTING DATA
df = pd.DataFrame.from_csv('batting3.csv', header=0, index_col=0)
df=df.rename(columns = {'Unnamed: 1': 'Year'})
df = df[df.Pos != 0] # REMOVING PITCHERS AND DHS
df = df[df.Year != 2017] # 2017 IS ON SEPARATE SCALE
df.Pos = df.Pos - 1 # INDEX POSITIONS AT 0
dfs = df[df.PA >= 100] # REMOVE PLAYERS WITH <= 100 PLATE APPEARANCES
data = dfs.as_matrix(columns = ['Year', 'Age', 'Lg', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'MapPos'])

np.random.shuffle(data)

params = data.shape[1] - 1

X = data[:int(data.shape[0]*.8), :params] #TRAINING
y = np.int64(data[:int(data.shape[0]*.8), params]) # MUST BE INTS

X_val = data[int(data.shape[0]*.8):, :params] #TEST
y_val = np.int64(data[int(data.shape[0]*.8):, params]) # MUST BE INTS

X = np.float64(X) # SO THAT PROCESSING POSSIBLE
X -= np.mean(X, axis = 0) # MEAN SUBTRACT
X /= np.std(X, axis = 0) # NORMALIZE

X_val = np.float64(X_val) # SO THAT PROCESSING POSSIBLE
X_val -= np.mean(X_val, axis = 0) # MEAN SUBTRACT
X_val /= np.std(X_val, axis = 0) # NORMALIZE

data = np.float64(data) # SO THAT PROCESSING POSSIBLE
data -= np.mean(data, axis = 0) # MEAN SUBTRACT
data /= np.std(data, axis = 0) # NORMALIZE

# initialize parameters randomly
D = params
K = len(set(data[:, params]))
h = 25 # size of hidden layer
W = 0.01 * np.random.randn(D,h)
b = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K))

# some hyperparameters
step_size = 1e-2
reg = 1e-3 # regularization strength

# gradient descent loop
num_examples = X.shape[0]

losses, accuracies, top3_acc = [], [], [] # EMPTY LISTS

start = time.time()

for i in range(1, 10000):

    if i%5000 == 0 and i>1:
          #step_size = np.float16(step_size * .9)
        print("learning rate: ", step_size)

      # evaluate class scores, [N x K]
    hidden_layer = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
    scores = np.dot(hidden_layer, W2) + b2

      # compute the class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

      # compute the loss: average cross-entropy loss and regularization
    corect_logprobs = -np.log(probs[range(num_examples),y])
    data_loss = np.sum(corect_logprobs)/num_examples
    reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
    loss = data_loss + reg_loss

    if i%10 == 0:
        predicted_class = np.argmax(scores, axis=1)
        accuracies.append(np.mean(predicted_class == y))
    if i%1000 == 0:
        print("iteration: %d: loss %f" % (i, loss), " time: {}".format(round(time.time() - start)),
                ' training acc: %.4f' % (np.mean(predicted_class == y)),
                ' top3: ', round(np.mean(top3s), 2))

    losses.append(loss)

    sort_scores = np.argsort(scores)
    top3 = np.fliplr(sort_scores)[:, :3]
    top3s = np.zeros(top3.shape[0])
    for i in range(top3.shape[0]):
        if y[i] in top3[i]:
            top3s[i] = 1
    top3_acc.append(np.mean(top3s))

      # compute the gradient on scores
    dscores = probs
    dscores[range(num_examples),y] -= 1
    dscores /= num_examples

      # backpropate the gradient to the parameters
      # first  backprop into parameters W2 and b2
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
      # next backprop into hidden layer
    dhidden = np.dot(dscores, W2.T)
      # backprop the ReLU non-linearity
    dhidden[hidden_layer <= 0] = 0
      # finally into W,b
    dW = np.dot(X.T, dhidden)
    db = np.sum(dhidden, axis=0, keepdims=True)

      # add regularization gradient contribution
    dW2 += reg * W2
    dW += reg * W

      # perform a parameter update
    W += -step_size * dW
    b += -step_size * db
    W2 += -step_size * dW2
    b2 += -step_size * db2

plt.plot(losses)
plt.plot(top3_acc)
plt.plot(accuracies)
plt.xlabel("epochs")
plt.ylabel("training accuracy (orange and green) and loss (blue)")
plt.savefig("mappos")
plt.show()
hidden_layer = np.maximum(0, np.dot(X_val, W) + b)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print('test accuracy: %.3f' % (np.mean(predicted_class == y_val)))

'''
sort_scores = np.argsort(scores)
top3 = np.fliplr(sort_scores)[:, :3]
top3s = np.zeros(top3.shape[0])
for i in range(top3.shape[0]):
    if y[i] in top3[i]:
        top3s[i] = 1
print('top3 test accuracy: %.3f' % np.mean(top3s))
'''

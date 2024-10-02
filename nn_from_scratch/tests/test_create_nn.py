import numpy as np
import matplotlib.pyplot as plt
from DimLib.auxialry_classwork import eval_numerical_gradient, eval_numerical_gradient_array, rel_error
import torch

from DimLib.nn_from_scratch import Linear, SoftMax, LogSoftMax, BatchNormalization, ChannelwiseScaling, Dropout, Sequential, \
                                    LeakyReLU, ELU, SoftPlus, ReLU, \
                                    ClassNLLCriterion, ClassNLLCriterionUnstable, \
                                    adam_optimizer, sgd_momentum
from time import time, sleep
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
 
def one_hot(y):
    N = 10 #for MNIST
    y_new = np.zeros((y.shape[0], N))
    y_new[np.arange(y.shape[0]), y] = 1
    return y_new

import mnist #perhapse chenge directory for import
X_train, y_train, X_val, y_val, X_test, y_test = mnist.load_dataset(flatten=True)

y_val = one_hot(y_val)
y_train = one_hot(y_train)
y_test = one_hot(y_test)

net = Sequential()
in_hidden = 64 #slighlty worse than for 64
net.add(Linear(28**2, in_hidden))

# net.add(BatchNormalization(alpha=0.9)) #works with alpha = 0.1 lol what???
# net.add(ChannelwiseScaling(in_hidden))


# net.add(LeakyReLU())
net.add(ReLU())
# net.add(ELU())
# net.add(SoftPlus())

net.add(Linear(in_hidden, 10))
net.add(LogSoftMax())
criterion = ClassNLLCriterion()

optimizer_config = {'learning_rate' : 4e-4, 'momentum': 0.9, "beta1": 0.9, "beta2": 0.999, 'epsilon': 1e-8}
optimizer_state = {}

# Looping params
n_epoch = 20
batch_size = 128

def get_batches(dataset, batch_size):
    X, Y = dataset
    n_samples = X.shape[0]
        
    # Shuffle at the start of epoch
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        
        batch_idx = indices[start:end]
    
        yield X[batch_idx], Y[batch_idx]


loss_history = []

for i in range(n_epoch):
    for x_batch, y_batch in get_batches((X_train, y_train), batch_size):
        
        net.zeroGradParameters()
        
        # Forward
        predictions = net.forward(x_batch)
        loss = criterion.forward(predictions, y_batch)
    
        # Backward
        dp = criterion.backward(predictions, y_batch)
        net.backward(x_batch, dp)
        
        # Update weights
        adam_optimizer(net.getParameters(), 
                     net.getGradParameters(), 
                     optimizer_config,
                     optimizer_state)      
        loss_history.append(loss)

    # Visualize
    
    if i == 19:
        display.clear_output(wait=True)
        plt.figure(figsize=(8, 6))
            
        plt.title("Training loss")
        plt.xlabel("#iteration")
        plt.ylabel("loss")
        plt.plot(np.log(loss_history), 'b')
        plt.show()
    print('Current loss: %f' % loss, np.log(loss))  

def encode_onehot(y):
    return np.argmax(y, axis=1)


#torch implementation
import torch.nn as nn
# import torch.functional as nn


model = nn.Sequential()
model.add_module("l1", nn.Linear(784, 64))
model.add_module("l2", nn.ReLU())
model.add_module("l5", nn.Linear(64, 10))
model.add_module("l8", nn.LogSoftmax())


opt = torch.optim.Adam(model.parameters(), lr=0.001)
history = []
for i in range(n_epoch):
    for x_batch, y_batch in get_batches((X_train, y_train), batch_size):    # sample 256 random images
        # ix = np.random.randint(0, len(X_train), 256)
        x_batch = torch.tensor(x_batch, dtype=torch.float32)
        y_batch = torch.tensor(y_batch, dtype=torch.float32)

        # predict probabilities
        y_predicted = model(x_batch) #[:, 0]  # YOUR CODE HERE

        # compute loss, just like before
        # loss = torch.mean(-(y_batch * torch.log(y_predicted) + (1-y_batch)*torch.log(1-y_predicted)).mean()) #lol wtf
        # print(y_predicted.shape)
        loss = nn.functional.cross_entropy(y_predicted, y_batch)
        loss.backward()
        opt.step()
        opt.zero_grad()

        history.append(loss.data.numpy())
        if i % 10 == 0:
            print("step #%i | mean loss = %.3f" % (i, np.mean(history[-10:])))


X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = encode_onehot(y_test)
y_test = torch.tensor(y_test, dtype=torch.long)
pred_test = model(X_test) 
pred_test = torch.argmax(pred_test, dim=1)

accuracy = torch.sum(pred_test == y_test) / len(y_test)
print("Test accuracy: %.5f" % accuracy)
assert accuracy > 0.95, "try training longer"
print("Great job!")




#eval on val
net.evaluate()
print(net.training)


predictions = net.forward(X_val)
loss = criterion.forward(predictions, y_val)
print(loss, np.log(loss))

pred = encode_onehot(predictions)
y_val = encode_onehot(y_val)
good_pred = np.sum(y_val==pred)
print("Accuracy:", good_pred / y_val.shape[0])


# #evaluation on test:
# net.evaluate()
# print(net.training)


# predictions = net.forward(X_test)
# loss = criterion.forward(predictions, y_test)
# print(loss)

# pred = encode_onehot(predictions)
# y_test = encode_onehot(y_test)
# good_pred = np.sum(y_test==pred)
# print("Accuracy:", good_pred / y_test.shape[0])

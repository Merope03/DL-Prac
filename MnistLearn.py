from tensorflow.keras.datasets import mnist  # type: ignore
import numpy as np
import matplotlib.pyplot as plt

from optimizer import *


from TwoLayerNetPrac import *
from Layer import *

optimizer = Momentum(lr = 0.01)

# Reading variables containing the data
from mnist import load_mnist

(x_train, y_train), (x_test, y_test) = \
    load_mnist(normalize = True, one_hot_label= False)

# 데이터 로드


x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28*28)

batch_size = 200
epoch = 25
iter_per_epoch = 60000 // batch_size

accuracies = []
losses = []

Network = MyNet(input_size = 28*28, hidden_size = 100, hidden_size2 = 50, output_size = 10)

for i in range(iter_per_epoch * epoch) :
    batch_mask = np.random.choice(60000, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]
    Network.backward(x_batch, y_batch)
    

    optimizer.update(Network.params, Network.grads)



    if i % (iter_per_epoch) == 0 :
        test_accuracy = Network.accuracy(x_test, y_test)
        test_loss = Network.loss(x_test, y_test)
        print("iter : ", i, "epoch : ", i // iter_per_epoch, 'testaccuracy : ', test_accuracy, "test_loss : ", test_loss)
        accuracies.append(test_accuracy)
        losses.append(test_loss)

plt.plot(accuracies)
plt.ylim(0.9, 0.99)
plt.show()
plt.plot(losses)
plt.show()

np.savez('mynet_params.npz', *Network.params)
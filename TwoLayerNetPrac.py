from Layer import *

class MyNet() :
    def __init__(self, input_size, hidden_size, hidden_size2, output_size) :
        W1, W2, W3 = np.random.randn(input_size, hidden_size)*0.01, np.random.randn(hidden_size, hidden_size2)*0.01, np.random.randn(hidden_size2, output_size)*0.01
        b1, b2, b3 = np.zeros(hidden_size), np.zeros(hidden_size2), np.zeros(output_size)
        self.params = [W1, b1, W2, b2, W3, b3]
        self.layers = [Affine(W1, b1), Relu(), Affine(W2, b2), Relu(), Affine(W3, b3)]
        self.grads = [0,0,0,0,0,0]
        self.lastlayer = SoftmaxAndCrossEntropy()

    def predict(self, x : np.ndarray) -> np.ndarray :
        for layer in self.layers :
            x = layer.forward(x)

        self.layers[0].W, self.layers[0].b = self.params[0], self.params[1]
        self.layers[2].W, self.layers[2].b = self.params[2], self.params[3]
        self.layers[4].W, self.layers[4].b = self.params[4], self.params[5]

        return x

    def loss(self, x : np.ndarray, y : np.ndarray) -> float :
        z = self.predict(x)
        loss = self.lastlayer.forward(z, y)
        return loss

    def backward(self, x : np.ndarray, y : np.ndarray) :
        _ = self.loss(x, y)
        dout = self.lastlayer.backward()
        for layer in self.layers[::-1] :
            dout = layer.backward(dout)
        self.grads[0] = self.layers[0].dW
        self.grads[1] = self.layers[0].db
        self.grads[2] = self.layers[2].dW
        self.grads[3] = self.layers[2].db
        self.grads[4] = self.layers[4].dW
        self.grads[5] = self.layers[4].db

        return

    def accuracy(self, x : np.ndarray, y : np.ndarray) :
        z = self.predict(x)
        return np.sum((np.argmax(z, axis = 1) == y)) / y.size
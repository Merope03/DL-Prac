from BasicFunctions import *

class Affine() :
    def __init__(self, W : np.ndarray, b : np.ndarray) :
        '''W : weight, b : bias'''
        self.W = W
        self.X = None   # W, X는 역전파 시 필요 --> 저장할 공간 만듦듦
        self.b = b      # W와 b가 이 클래스를 결정하니 저장해 두자.
        self.dW = None
        self.db = None


    def forward(self, X : np.ndarray) -> np.ndarray :
        '''
        Affine 클래스의 순전파를 실행한다.
        반환 : XW+b
        '''
        self.X = X    # 역전파 시 사용하기 위해 저장.
        return np.dot(X,self.W)+self.b

    def backward(self, dout : np.ndarray) -> np.ndarray :
        '''
        Affine 클래스의 역전파를 실행한다.
        순전파를 실행했을 때의 X,W,b가 저장되어 있다.
        반환 : dX
        입력값은 Scalar Loss Function L에 대한 Affine 계층의 결과값(Y = XW+b)의 미분이다.
        '''
        dX = np.dot(dout, self.W.transpose())
        dW = np.dot(self.X.transpose(), dout)
        db = np.sum(dout, axis = 0)

        self.dW = dW
        self.db = db

        return dX


class Relu() :
    def __init__(self) :
        self.X = None # 입력값을 기억해야 BackProp 가능

    def forward(self, X : np.ndarray) -> np.ndarray :
        '''Ex.
        [
        [1,1,-1,-1],
        [-2,0,1,1]
        ]
        -->
        [
        [1,1,0,0],
        [0,0,1,1]
        ]
        '''
        self.X = X
        X_mask = (X >0)
        return X_mask*X

    def backward(self, dout : np.ndarray) -> np.ndarray :
        return (self.X > 0) * dout


class SoftmaxAndCrossEntropy :
    def __init__(self) :
        '''y는 label data.
        Ex.
        [2,0,3,1,3]
        '''
        self.y = None
        self.Z : np.ndarray = None
        self.batch_size = None
        return

    def forward(self, Z : np.ndarray, y : np.ndarray) -> np.ndarray :
        '''
        Loss 산출
        '''
        self.y = y
        self.Z = Z
        softmax_z = Softmax(Z)
        CrossEntropyLoss = CrossEntropy(softmax_z, y)
        self.batch_size = Z.shape[0]
        return CrossEntropyLoss

    def backward(self) -> np.ndarray :
        return (Softmax(self.Z) - np.eye(self.Z.shape[1])[self.y]) / self.batch_size

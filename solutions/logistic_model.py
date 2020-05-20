# pure python version
def _cross_entropy(X, Y, W, b):
    return -sum(
        -z*(1-y) - np.log(1+np.exp(-z)) 
        for z, y 
        in zip(logodds(X, W, b), Y)
    ) / df.shape[0]
        
def cross_entropy(X, Y, W, b):
    Z = logodds(X, W, b) 
    logliks = -Z*(1 - Y) - np.log(1 + np.exp(-Z))
    return -logliks.mean()

def gradient_descent(X, Y, W, b, η=0.01):
    Yhat = expit(logodds(X, W, b))
    δ = Yhat - Y
    dW = X.T @ δ / δ.shape[0]
    db = δ.mean()
    assert dW.shape == W.shape
    return W - η * dW, b - η * db
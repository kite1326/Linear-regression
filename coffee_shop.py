import numpy as np
import pandas as pd

df = pd.read_csv(r"C:\Users\Kiet\Downloads\coffee_shop_revenue.csv")
matrix = df.to_numpy()

X = matrix[:, :matrix.shape[1] -1]
y = matrix[:, matrix.shape[1] - 1].reshape(-1, 1)

X[:,0]/=500
X[:,1]/=10
X[:,2]/=18
X[:,3]/=15
X[:,4]/=500
X[:,5]/=1000

one = np.ones((X.shape[0], 1))
Xbar = np.column_stack((one, X))


#Cách 1: Dùng đạo hàm = 0
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)

#Cách 2: Dùng Gradien Descent
def grad(w):
    N = Xbar.shape[0]
    return 1/N * Xbar.T.dot(Xbar.dot(w) - y)
def loss(w):
    N = Xbar.shape[0]
    return .5/N*np.linalg.norm(y - Xbar.dot(w), 2)**2;
def GD(w_init, grad, eta):
    w = w_init
    for it in range(1000000):
        w_new = w - eta*grad(w)
        if np.linalg.norm(grad(w_new)) < 1e-3:
            break 
        w = w_new
    return (w, it)
w_init = np.array([[-1517],[ 2783],[ 2432],[-4],[-34 ],[777],[24]])
(w1, it1) = GD(w_init, grad, 0.0001)
print('Trong so w = ', w1, ',\nsau %d vong lap' %(it1+1))


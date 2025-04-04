#Mô phỏng mô hình dự đoán cân nặng dựa theo chiều cao


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Chiều cao
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
#cân nặng
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T


#Tạo ma trận Xbar gồm ma trận cột 1 nối với X
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

#Cách 1: Dùng đạo hàm = 0
#Tính w = (Xbar.T*Xbar) giả nghịch nhân với (Xbar.T*y)
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)

#Cách 2: Dùng Gradien Descent
#Hàm đạo hàm
def grad(w):
    N = Xbar.shape[0]
    return 1/N * Xbar.T.dot(Xbar.dot(w) - y)
#cost function
def cost(w):
    N = Xbar.shape[0]
    return .5/N*np.linalg.norm(y - Xbar.dot(w), 2)**2;
#Hàm Gradient Descent
def GD(w_init, grad, eta):
    w = [w_init]
    for it in range(1000):
        w_new = w[-1] - eta*grad(w[-1])
        if np.linalg.norm(grad(w_new)) < 1e-3:  # Điều kiện dừng khi gradient đủ nhỏ
            break 
        w.append(w_new)
    return (w, it)
w_init = np.array([[-33], [0.5]])
(w1, it1) = GD(w_init, grad, 0.000000000000001)
print('Solution found by GD: w = ', w1[-1], ',\nafter %d iterations.' %(it1+1))

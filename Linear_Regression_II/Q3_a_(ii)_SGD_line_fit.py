import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics import mean_squared_error
from copy import deepcopy

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro', animated=True)
ln.set_data([1,3,6],[6,10,16])
x = [1,3,6]
y = [6,10,16]
line, = ax.plot(x,y)
def sgd(X,Y,alpha,theta):
    
    ch = random.randint(0,len(X)-1)
    x = X[ch]
    y = Y[ch] 
    theta = theta - alpha*gradient(x,y,theta)
    return theta

def gradient(x,y,theta): 
    
    grad = np.zeros(len(theta))
    err = error(x,y,theta)
    for j in range(len(theta)):
        grad[j] = 2*x[j]*err
    return grad

def error(x,y,theta):
    err = np.dot(x,theta)-y
    return err

def score(test_x,test_y,theta):
    predict = []
    for i in test_x:
        prediction = np.dot(i,theta)
        predict.append(prediction)
    predict = np.asarray(predict)
    error = mean_squared_error(predict,test_y)
    return error**0.5

X = np.array([[1,1],[1,3],[1,6]])
Y = np.array([6,10,16])
theta_list = []
mse = []
theta_old = np.zeros(len(X[0]))
alpha = 0.01

for i in range(1000):
    theta_new = sgd(X,Y,alpha,theta_old)
    theta_old = deepcopy(theta_new)
    theta_list.append(theta_new)
    mse.append(score(X,Y,theta_new))


def init():
    line.set_ydata([np.nan]*3)
    return line,ln,

def animate(i):
    frame = theta_list[i]
    x_data = np.array([1,3,6])
    y_data = np.array([round(frame[0],1)+round(frame[1],1)*i for i in x_data])
    line.set_ydata(y_data)
    plt.scatter([1,3,6],[6,10,16],c = 'r')
    plt.title('Iteration'+str(i)+' MSE '+ str(round(mse[i],1))+' theta0 '+str(round(frame[0],1))+' theta1 '+str(round(frame[1],1)))

ani = FuncAnimation(fig, animate, frames=np.arange(500),interval = 1,
                    init_func=init, blit=False)

plt.show()

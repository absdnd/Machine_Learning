import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics import mean_squared_error
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from copy import deepcopy 


fig, ax = plt.subplots()
xdata, ydata = np.arange(-1.0,9.0,1),np.arange(-1.0,9.0,1)
thi = np.array([0,0])

ln, = plt.plot([], [], 'ro', animated=True)

def f(th0,th1):

    residue = (6-th0-th1)**2+(10-th0-3*th1)**2 + (16-th0-6*th1)**2
    return residue

def update_theta(X,Y,theta,ch): 
    rows = len(X)
    coef = 0.
    theta[ch] = 0.
    num = 0.
    den = 0.
    for i in range(len(X)):
        den += 2.*X[i,ch]**2
        num += 2.*X[i,ch]*error(X[i],Y[i],theta)
    theta[ch] = -1.0*num/den
    return theta

def coordinateDescentRegression(X,Y,theta): 
    
    
    ch = random.randint(0,len(X[0])-1)
        
    theta = update_theta(X,Y,theta,ch)
    return theta


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
theta_old = np.zeros(2)


for i in range(100):
    theta_new = coordinateDescentRegression(X,Y,theta_old)
    theta_old = deepcopy(theta_new)
    theta_list.append(theta_new)
    mse.append(score(X,Y,theta_new))
    
def init():
    X_,Y_ = np.meshgrid(xdata,ydata)
    Z = f(X_,Y_)
    CS = ax.contour(X_,Y_,Z)
    ax.set_xlim(-1,9)
    ax.set_ylim(-1,9)
    return ax,

def animate(i):
    theta = theta_list[i]
    theta_next = theta_list[i+1]

    ln.set_data(int(theta[0]),int(theta[1]))
    ax.arrow(theta[0],theta[1],theta_next[0]-theta[0],theta_next[1]-theta[1],head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.set_title('Iteration'+str(i)+' MSE '+ str(round(mse[i],1))+' theta0 '+str(round(theta[0],1))+' theta1 '+str(round(theta[1],1)))
    return ln,ax
ani = FuncAnimation(fig, animate, frames=np.arange(99),interval = 200,
                    init_func=init, blit=False)

plt.show()

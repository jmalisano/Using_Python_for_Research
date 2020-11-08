import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

#generate synthetic data
n = 100
beta_0 = 5
beta_1 = 2
np.random.seed(1)
x = ss.uniform.rvs(size = n)*10 #100 random variables between 0 and 10
y = beta_0+beta_1*x + ss.norm.rvs(loc=0, scale=1, size=n) #y=mx+c+noise. It is assumed that y=mx+c is the 

plt.figure()
plt.plot(x,y, "o", ms=5)
xx = np.array([0, 10]) #lowest and highest x
plt.plot(xx, beta_0 + beta_1 * xx)

def compute_rss(y_estimate, y):
  return sum(np.power(y-y_estimate, 2))

def estimate_y(x, b_0, b_1):
  return b_0 + b_1 * x

#rss = compute_rss(estimate_y(x, beta_0, beta_1), y)


#we are attempting to estimate least squares
#in this we are pretendeding we already know beta0
rss = []
slopes = np.arange(-10, 15, 0.01)
for slope in slopes:
    rss.append(np.sum((y - beta_0 - slope*x)**2)) #y - beta_0 - slope*x gives us difference between predicted y and observed y, doing for a range of slopes

ind_min = np.argmin(rss) #findex the index of the rss array that has  the min value

#print("estimate for slope:", slopes[ind_min])
    
#an easier method to find the slope is as follows
import statsmodels.api as sm
mod = sm.OLS(y,x)   #x are the predictor values
est = mod.fit()
print(est.summary())

X = sm.add_constant(x) #same as x but include colum of 1s
mod=sm.OLS(y,X)
est = mod.fit()
print(est.summary())

n = 500
beta_0 = 5
beta_1 = 2
beta_2 = -1
np.random.seed(1)

#generates random scatter of x1 and x2
x_1 = 10 * ss.uniform.rvs(size=n)
x_2 = 10 * ss.uniform.rvs(size=n)

#outcome
y = beta_0 + beta_1*x_1 + beta_2*x_2 + ss.norm.rvs(loc=0, scale=1, size=n)

#take x1 and x2 and stack as columns in a matrix
X = np.stack([x_1, x_2], axis=1)

from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], y, c=y)

from sklearn.linear_model import LinearRegression
lm = LinearRegression(fit_intercept=True)
lm.fit(X,y)
print(f"intercept: {lm.intercept_}")
print(f"coefficients [x1,x2]: {lm.coef_}")

#predict value of outcome for some X
X_0 = np.array([2,4]) #take x1=2, x2=4

lm.predict(X_0.reshape(1, -1))
lm.score(X, y)  #finds R2, takes input X, generates prediction of y, compares outcome with actual y

#ASSESSING MODEL ACCURACCY
#Mean squared  error (MSE) is most common way to assess accuraccy

#how to split data to training and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=1)
lm = LinearRegression(fit_intercept=True) #creates a model object
lm.fit(X_train, y_train) #fits training data

wlm.score(X_test, y_test) #takes input X data (test), then compares outputs to test Ys


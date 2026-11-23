import numpy as np
import matplotlib.pyplot as plt

def rmse(t: np.ndarray, tp: np.ndarray):
    return float(np.sqrt(np.mean(np.abs(t - tp) **2)))
    
# load data
train_data = np.loadtxt("boston_train.csv", delimiter=",")
test_data = np.loadtxt("boston_test.csv", delimiter=",")
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]

# make sure that we have N-dimensional Numpy arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1))
t_test = t_test.reshape((len(t_test), 1))
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

# (a) compute mean of prices on training set
mean_price = t_train.mean()
print("a - mean: ", mean_price)

# (b) RMSE function # def rmse(t, tp):
# here we need an array with duplicates of mean_price, otherwise np just scales
# the each number in the array like (array - number) = not good
t_test_pred = np.full_like(t_test, mean_price)
error = rmse(t_test, t_test_pred)
print("b - rmse: ", error)

# (c) visualization of results
# ravel flattens the (n, 1) arrays for plotting
plt.scatter(t_test.ravel(), t_test_pred.ravel())
plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")
plt.show()
import numpy as np
import pandas
import linreg
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
#print("Number of training instances: %i" % X_train.shape[0])
#print("Number of test instances: %i" % X_test.shape[0])
#print("Number of features: %i" % X_train.shape[1])

# (b) fit linear regression using only the first feature
model_single = linreg.LinearRegression()
model_single.fit(X_train[:,0], t_train)
print("crim only weights:\n", model_single.w)

# (c) fit linear regression model using all features
model_all = linreg.LinearRegression()
model_all.fit(X_train, t_train)
print("every parameter weights:\n", model_all.w)

# (d) evaluation of results
# compute test predictions
pred_single = model_single.predict(X_test[:, 0])
pred_all = model_all.predict(X_test)
#print("single predictions:\n", pred_single)
#print("all predictions:\n", pred_all)

# compute the error for single and all
single_error = rmse(t_test, pred_single)
all_error = rmse(t_test, pred_all)
print("single error: ", single_error)
print("all error: ", all_error)

# plot actual prices vs predicted prices
plt.scatter(t_test.ravel(), pred_single.ravel())
plt.xlim(0, 55)
plt.ylim(-20, 30)  # single-feature CRIM predictions can go negative for high crime
plt.title("Only CRIM")
plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")
plt.figure()

plt.scatter(t_test.ravel(), pred_all.ravel())
plt.title("All parameters")
plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")
plt.show()
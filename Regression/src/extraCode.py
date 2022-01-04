import numpy as np
from polynomial_regression import PolynomialRegression
from generate_regression_data import generate_regression_data
from metrics import mean_squared_error

try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


if __name__ == '__main__':
    #1a

    x, y = generate_regression_data(4, 100, 0.1)
    indexes = np.random.choice(range(100), 50, replace=False)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    pList = []
    mse_train = []
    mse_test = []
    for i in range(100):
        if i in indexes:
            x_train.append(x[i])
            y_train.append(y[i])
        else:
            x_test.append(x[i])
            y_test.append(y[i])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    for i in range(9):
        p = PolynomialRegression(i)
        p.fit(x_train, y_train)
        p.visualize(x_train, y_train)
        p.visualize(x_test, y_test)

        yhattrain = p.predict(x_train)
        mse_train.append((mean_squared_error(y_train, yhattrain)))
        yhattest = p.predict(x_test)
        mse_test.append(mean_squared_error(y_test, yhattest))
        pList.append(p)

    plt.figure()
    plt.plot(range(9), [np.log(mse_train[i]) for i in range(9)], label="training error")
    plt.plot(range(9), [np.log(mse_test[i]) for i in range(9)], label="testing error")
    plt.title("Training and Testing Errors vs. Degree")
    plt.xlabel('degree')
    plt.ylabel("log of error")
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/ashlee/Videos/mseError50.png')

    #1b

    minTrainDegree = mse_train.index(min(mse_train))
    minTestDegree = mse_test.index(min(mse_test))

    plt.clf()
    plt.figure()
    plt.scatter(x_train, y_train)
    plt.plot(np.sort(pList[minTrainDegree].x), pList[minTrainDegree].h,
             label="min train err curve, degree = " + str(minTrainDegree))
    plt.plot(np.sort(pList[minTestDegree].x),
             pList[minTestDegree].h,
             label="min test err curve, degree = " + str(minTestDegree))
    plt.title("Min. Training and Testing Errors Curves")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.savefig('/home/ashlee/Videos//minError50.png')













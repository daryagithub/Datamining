import pandas
import numpy
import seaborn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tools import eval_measures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm


class Ecommerce:

    def __init__(self, fileName):
        self.data = pandas.read_csv(fileName)
        self.xTrain = None
        self.xTest = None
        self.yTrain = None
        self.yTest = None
        self.yTestPredicted = None
        self.target = None
        self.attributes = None
        self.model = None


    def showHeatMap(self):
        corr = self.data.corr()
        size = len(self.data.columns)
        plt.figure(figsize=(size, size))
        ax = seaborn.heatmap(corr, vmin=0, vmax=1, center=0, square=True, cmap="RdYlGn")
        plt.savefig('Heat Map.png')
        plt.show()

    def showpairPlot(self):
        size = len(self.data.columns)
        plt.figure(figsize=(size, size))
        ax = seaborn.pairplot(self.data)
        plt.savefig('Pair Plot.png')
        plt.show()

    def splitData(self, attributes, target, splitRate):
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(
            self.data[attributes], self.data[target], test_size=splitRate, random_state=42
        )
        self.attributes = attributes
        self.target = target

    def fitLinearRegression(self):
        fitter = LinearRegression()
        self.model = fitter
        fitter.fit(X=self.xTrain, y=self.yTrain)
        predicted = fitter.predict(X=self.xTrain)
        self.yTestPredicted = fitter.predict(X=self.xTest)
        intercept = fitter.intercept_
        coefs = fitter.coef_
        print("Linear Regression Results:")
        print("Target is: " + str(self.target))
        print("InterCept: " + str(intercept))
        print("Slop Coefficient For Each Feature:")
        for i in range(len(coefs)):
            print(self.attributes[i] + ': ' + str(coefs[i]))
        MSE = mean_squared_error(y_true=self.yTrain, y_pred=predicted)
        print("MSE Train: " + str(MSE))
        RMSE = eval_measures.rmse(x1=self.yTrain, x2=predicted)
        print("RMSE Train: " + str(RMSE))

    def crossValidateBykFold(self, k):
        fitter = LinearRegression()
        accuracy = cross_val_score(estimator=fitter, X=self.xTrain, y=self.yTrain, cv=k)
        print("Accuracy of Model with Cross Validation is:", numpy.round(accuracy.mean() * 100, 2), str("%"))

    def evaluateModel(self):
        MSE = mean_squared_error(y_true=self.yTest, y_pred=self.yTestPredicted)
        print("MSE Test: " + str(MSE))
        RMSE = eval_measures.rmse(x1=self.yTest, x2=self.yTestPredicted)
        print("RMSE Test: " + str(RMSE))


def program():
    model = Ecommerce(fileName='Ecommerce Customers.csv')
    # model.showHeatMap()
    # model.showpairPlot()
    model.splitData(attributes=['Avg. Session Length',
                                'Time on App',
                                'Time on Website',
                                'Length of Membership'],
                    target='Yearly Amount Spent',
                    splitRate=0.25)
    model.fitLinearRegression()
    model.crossValidateBykFold(k=10)
    model.evaluateModel()


program()

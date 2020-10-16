import pandas
import numpy
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tools import eval_measures


class Preprocessing:

    def __init__(self, fileName, skipRows, numberOfRows):
        self.data = pandas.read_csv(fileName, skiprows=skipRows, nrows=numberOfRows, index_col=0)

    def handleMissingValuesByArbitraryValue(self, value):
        self.data.fillna(value, inplace=True)

    def handleMissingValuesByMean(self):
        self.data.fillna(self.data.mean(), inplace=True)
        print()

    def handleMissingValuesByMode(self):
        self.data.fillna(self.data.mode(), inplace=True)

    def handleMissingValuesByMedian(self):
        self.data.fillna(self.data.median(), inplace=True)

    def encodeDataByOneHotEncoding(self, columns):
        for column in columns:
            encodedData = pandas.get_dummies(self.data[column], prefix=column)
            self.data = self.data.join(encodedData)
        self.data = self.data.drop(columns=columns)

    def encodeDataByLabelEncoding(self, columns):
        for column in columns:
            columnData = numpy.array(self.data[column]).reshape(-1, 1)
            encoder = LabelEncoder()
            encodedData = encoder.fit_transform(columnData)
            self.data[column] = encodedData

    def scaleDataByMinMaxScaler(self, columns):
        for column in columns:
            minMaxScaler = MinMaxScaler()
            unscaledData = numpy.array(self.data[column]).reshape(-1, 1)
            scaledData = minMaxScaler.fit_transform(unscaledData)
            self.data[column] = scaledData

    def scaleDataByStandardScaler(self, columns):
        for column in columns:
            standardScaler = StandardScaler()
            unscaledData = numpy.array(self.data[column]).reshape(-1, 1)
            scaledData = standardScaler.fit_transform(unscaledData)
            self.data[column] = scaledData

    def scaleDataByRobustScaler(self, columns):
        for column in columns:
            robustScaler = RobustScaler()
            unscaledData = numpy.array(self.data[column]).reshape(-1, 1)
            scaledData = robustScaler.fit_transform(unscaledData)
            self.data[column] = scaledData

    def scaleDataByNormalizer(self, columns):
        for column in columns:
            normalizer = Normalizer
            unscaledData = numpy.array(self.data[column]).reshape(-1, 1)
            scaledData = normalizer.transform(unscaledData)
            self.data[column] = scaledData

    def removeOutliersByNormalDist(self, columns):
        for column in columns:
            values = numpy.array(self.data[column]).reshape(-1, 1)
            mean = numpy.mean(values)
            std = numpy.std(values)
            upperBound = mean + 3 * std
            lowerBound = mean - 3 * std
            self.data = self.data[lowerBound < self.data[column]]
            self.data = self.data[upperBound > self.data[column]]

    def removeOutliersByPercentage(self, columns, outlierPercentage):
        for column in columns:
            values = self.data[column]
            lowBound = values.quantile(outlierPercentage)
            highBound = values.quantile(1 - outlierPercentage)
            self.data = self.data[self.data[column].between(lowBound, highBound)]


    def fitLinearRegression(self, target):
        attributes = list(self.data.columns.values)
        attributes.pop(attributes.index(target))
        fitter = LinearRegression()
        fitter.fit(X=self.data[attributes], y=self.data[target])
        predicted = fitter.predict(X=self.data[attributes])
        intercept = fitter.intercept_
        coefs = fitter.coef_
        print("Linear Regression Results:")
        print("Target is: " + str(target))
        print("InterCept: " + str(intercept))
        print("Slop Coefficient For Each Feature:")
        for i in range(len(coefs)):
            print(attributes[i] + ': ' + str(coefs[i]))
        MSE = mean_squared_error(y_true=self.data[target], y_pred=predicted)
        print("MSE: " + str(MSE))
        RMSE = eval_measures.rmse(x1=self.data[target], x2=predicted)
        print("RMSE: " + str(RMSE))

    def fitPolyNomialRegression(self, attribute, target):
        coefs = numpy.polyfit(x=self.data[attribute], y=self.data[target], deg=2)
        coefeNames = ['a', 'b', 'c']
        print("Poly-Nomial Regression Results:")
        for i in range(len(coefeNames)):
            print(coefeNames[i] + ': ' + str(coefs[i]))



def program():
    model = Preprocessing(fileName="_Datapreprocessing.csv",
                          skipRows=2,
                          numberOfRows=16)
    model.handleMissingValuesByMean()
    # model.handleMissingValuesByMode()
    # model.handleMissingValuesByMedian()
    # model.handleMissingValuesByArbitraryValue(value=0)
    # model.encodeDataByOneHotEncoding(columns=[' CountryName', 'CountryCode', 'International Visitors'])
    model.removeOutliersByPercentage(columns=[' CountryName', 'CountryCode', 'International Visitors'], outlierPercentage=0.02)
    model.encodeDataByLabelEncoding(columns=[' CountryName', 'CountryCode', 'International Visitors'])
    model.scaleDataByMinMaxScaler(columns=['Population growth', 'Total population', 'Area (sq. km)', 'Coronavirus Cases'])
    model.fitLinearRegression(target='Coronavirus Cases')
    model.fitPolyNomialRegression(attribute='Total population', target='Coronavirus Cases')
    print()


program()
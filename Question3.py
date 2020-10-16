import pandas
import numpy
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


class Arrhythmia:

    def __init__(self, fileName):
        self.data = pandas.read_csv(fileName, header=None)
        self.data.rename(columns={279: 'target'}, inplace=True)
        self.xTrain = None
        self.xTest = None
        self.yTrain = None
        self.yTest = None
        self.yTestPredicted = None

    def handleMissingValuesByMean(self):
        self.data.replace('?', numpy.nan, inplace=True)
        self.data.fillna(0, inplace=True)

    def splitData(self, target, splitRate):
        attributes = list(self.data.columns.values)
        attributes.pop(attributes.index(target))
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(
            self.data[attributes], self.data[target], test_size=splitRate, random_state=42
        )

    def fitKNN(self, k):
        knnClassifier = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knnClassifier.fit(self.xTrain, self.yTrain)
        predictedTargetTrain = knnClassifier.predict(self.xTrain)
        predictedTargetTest = knnClassifier.predict(self.xTest)
        predictedTargetProbsTrain = knnClassifier.predict_proba(self.xTrain)
        predictedTargetProbsTest = knnClassifier.predict_proba(self.xTest)
        accuracyTrain = accuracy_score(y_true=self.yTrain, y_pred=predictedTargetTrain)
        accuracyTest = accuracy_score(y_true=self.yTest, y_pred=predictedTargetTest)
        precesionTrain = precision_score(y_true=self.yTrain, y_pred=predictedTargetTrain, zero_division=1, average='weighted')
        precesionTest = precision_score(y_true=self.yTest, y_pred=predictedTargetTest, zero_division=1, average='weighted')
        recallTrain = recall_score(y_true=self.yTrain, y_pred=predictedTargetTrain, zero_division=1, average='weighted')
        recallTest = recall_score(y_true=self.yTest, y_pred=predictedTargetTest, zero_division=1, average='weighted')
        targetNames = [i for i in range(17)]
        classificationReportTrain = classification_report(y_true=self.yTrain, y_pred=predictedTargetTrain, labels=targetNames, zero_division=1)
        classificationReportTest = classification_report(y_true=self.yTest, y_pred=predictedTargetTest, labels=targetNames, zero_division=1)
        confusionMatrixTrain = confusion_matrix(y_true=self.yTrain, y_pred=predictedTargetTrain)
        confusionMatrixTest = confusion_matrix(y_true=self.yTest, y_pred=predictedTargetTest)
        print("KNN Model Results:")
        print("K = " + str(k))
        print("Accuracy Train: " + str(numpy.round(accuracyTrain * 100, 2)))
        print("Accuracy Test: " + str(numpy.round(accuracyTest * 100, 2)))
        print("Precision Train: " + str(numpy.round(precesionTrain * 100, 2)))
        print("Precision Test: " + str(numpy.round(precesionTest * 100, 2)))
        print("Recaal Train: " + str(numpy.round(recallTrain * 100, 2)))
        print("Recaal Test: " + str(numpy.round(recallTest * 100, 2)))
        print("Classification Report Train:\n" + str(classificationReportTrain))
        print("Classification Report Test:\n" + str(classificationReportTest))
        print("Confusion Matrix Train:\n" + str(confusionMatrixTrain))
        print("Confusion Matrix Test:\n" + str(confusionMatrixTest))
        print('\n')


    def optimizeWithKFold(self):
        scores = []
        for i in range(1, 100):
            knnClassifier = KNeighborsClassifier(n_neighbors=i, metric='euclidean')
            knnClassifier.fit(self.xTrain, self.yTrain)
            predictedTargetTrain = knnClassifier.predict(self.xTrain)
            predictedTargetTest = knnClassifier.predict(self.xTest)
            accuracyTrain = accuracy_score(y_true=self.yTrain, y_pred=predictedTargetTrain)
            accuracyTest = accuracy_score(y_true=self.yTest, y_pred=predictedTargetTest)
            scores.append([accuracyTrain, accuracyTest])
        print("K\t\t\tAccuracy Train\t\t\t\t\tAccuracy Test")
        for i in range(1, 100):
            print(str(i), "\t\t\t", str(scores[i-1][0]), "\t\t\t", str(scores[i-1][1]))


def program():
    model = Arrhythmia(fileName='Arrhythmia.csv')
    model.handleMissingValuesByMean()
    model.splitData(target='target', splitRate=0.25)
    # model.fitKNN(k=1)
    # model.fitKNN(k=30)
    # model.optimizeWithKFold()
    model.fitKNN(k=5)


program()
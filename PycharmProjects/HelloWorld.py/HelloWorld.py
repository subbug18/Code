
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults.NNA"

names = ['X_Minimum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas', 'X_Maximum', 'X_Perimeter', 'Y_Perimeter',
         'Sum_of_Luminosity', 'Minimum_of_Luminosity', 'Maximum_of_Luminosity', 'Length_of_Conveyer',
         'TypeOfSteel_A300',
         'TypeOfSteel_A400', 'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index',
         'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index', 'Log_Y_Index',
         'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas', 'Pastry', 'Z_Scratch', 'K_Scatch', 'Stains',
         'Dirtiness', 'Bumps', 'Other_Faults']

dt = pandas.read_csv(url, names=names, dialect="excel-tab")

print(dt.shape)

# print(dt.describe())

validation_size = 0.20
seed = 7
arr = dt.values
X = arr[:, 0:28]
Y = arr[:, 28]

X_train, X_test, Y_train, Y_target = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                      random_state=seed)

models = []
models.append(('CART', DecisionTreeClassifier()))
# spot check algorithms
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
knnprediction = knn.predict(X_test)
# print(knnprediction)
print("****************")
# print(Y_target)

from sklearn.metrics import accuracy_score

print("KNN accuracy score: ", accuracy_score(Y_target, knnprediction))

Lnr = LinearDiscriminantAnalysis()
Lnr.fit(X_train, Y_train)
lnrpredictions = Lnr.predict(X_test)
print("Lnr accuracy score: ", accuracy_score(Y_target, lnrpredictions))

print(classification_report(Y_target, lnrpredictions))
pt = pandas.DataFrame(Y_target, lnrpredictions)
# pt = (Y_target, lnrpredictions)
# pt.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pt.hist()


import matplotlib.pyplot
import pylab

matplotlib.pyplot.scatter(Y_target, lnrpredictions)

matplotlib.pyplot.show()

matplotlib.pyplot.bar(Y_target, lnrpredictions)
matplotlib.pyplot.show()

matplotlib.pyplot.pie(Y_target, lnrpredictions)
matplotlib.pyplot.show()

matplotlib.pyplot.Circle(Y_target, lnrpredictions)
matplotlib.pyplot.show()

Python_machine_learning.py
Displaying
Python_machine_learning.py.
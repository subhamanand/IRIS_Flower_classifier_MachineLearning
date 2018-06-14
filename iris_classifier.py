import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
scoring="accuracy"
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []

print('\n\nAccuracy of different algorithms:\n')

for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


print('\n\nDataset:\n\n',dataset)

lr = LogisticRegression()
knn=KNeighborsClassifier()

knn.fit(X_train,Y_train)
lr.fit(X_train, Y_train)


predictionByLR=lr.predict(X_validation)
predictionByKNN=knn.predict(X_validation)

print('\n\n--------------------------------------------------')
print("Prediction by KNeighbours:",predictionByKNN)
print("Accuracy Score :",accuracy_score(Y_validation, predictionByKNN))

print('\n\n--------------------------------------------------')

print("Predicton by logistic regression",predictionByLR)
print("Accuracy Score:",accuracy_score(Y_validation, predictionByLR))


















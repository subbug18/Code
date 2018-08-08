from __future__ import print_function
from time import time
import logging
import matplotlib.pyplot as plt

from    sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import  PCA
from    sklearn.svm import SVC

print(__doc__)

#display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

lfw_people = fetch_lfw_people(min_faces_per_person=70,resize=0.4)

#introspect the image array to find the the shapes(for plotting)
n_samples, h, w = lfw_people.images.shape


X=lfw_people.data
n_features = X.shape[1]

Y=lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total data-set size:")
print("n_samples:%d" %n_samples)
print("n_features:%d" %n_features)
print("n_classes:%d" %n_classes)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)



n_components = 150

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


#!/usr/bin/python3
import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]

#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
# plt.show()


def apply_classifier(clftype):
    print('Now running {:s} classifier'.format(clftype))
    clf = classifier[clftype]
    t0 = time()
    clf.fit(features_train, labels_train)
    t1 = time()
    # print('Training took {:0.2f} seconds'.format(t1-t0))
    pred = clf.predict(features_test)
    t2 = time()
    # print('Predicting took {:0.2f} seconds'.format(t2-t1))
    accu = accuracy_score(labels_test, pred)
    print('Accuracy is : {:0.4f}'.format(accu))


classifier = {'K-neighbors': KNeighborsClassifier(n_neighbors=20, algorithm='kd_tree', leaf_size=5),
              'Random_Forest': RandomForestClassifier(n_estimators=30),
              'Adaboost': AdaBoostClassifier(n_estimators=100),
              'GaussianNB': GaussianNB(),
              'SVM': SVC(C=60000),
              'Decision Tree': DecisionTreeClassifier()}

# Highest score so far is achieved with SVC(C=60000) (0.9520)
# Next best performer are KNneighbors and Adaboost

for key in classifier:
    apply_classifier(key)
    plt.figure()
    prettyPicture(classifier[key], features_test, labels_test)
    plt.title('Using {:s} classifier'.format(key))

# Prevent figure from closing by itself
plt.show(block=True)
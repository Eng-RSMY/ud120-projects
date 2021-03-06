#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

with open("../final_project/final_project_dataset.pkl", "rb") as fin:
    data_dict = pickle.load(fin)

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list , sort_keys='../tools/python2_lesson13_keys.pkl')
labels, features = targetFeatureSplit(data)

### it's all yours from here forward!

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn import cross_validation

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels,
                                                                                             test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accu = accuracy_score(labels_test, pred)

print("Decicion tree prediction accuracy is : {:0.4f}".format(accu))
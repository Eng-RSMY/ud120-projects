#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys='../tools/python2_lesson13_keys.pkl')
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn import cross_validation

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels,
                                                                                             test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
n_poi = len([x for x in pred if x == 1])
accu = accuracy_score(labels_test, pred)
recal = recall_score(labels_test, pred)
prec = precision_score(labels_test, pred)
n_tru_pos = len([x for x, y in zip(pred, labels_test) if x == 1 and y == 1])


print("Decicion tree prediction accuracy is : {:0.4f}".format(accu))
print("Decicion tree prediction precision is : {:0.4f}".format(prec))
print("Decicion tree prediction recall is : {:0.4f}".format(recal))
print("There are {:d} pois in the total {:d} people in test set".format(n_poi, len(pred)))
print("There are {:d} true positive cases".format(n_tru_pos))
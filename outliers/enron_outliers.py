#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "rb") )
data_dict.pop('TOTAL', None)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

for item in data_dict:
    if float(data_dict[item]['salary']) > 1000000 and float(data_dict[item]['bonus']) > 5000000:
        print(item)


for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()



### your code below




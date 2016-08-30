#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

with open(r"../final_project/final_project_dataset.pkl", "rb") as fin:
    enron_data = pickle.load(fin)

# enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# count number of person of interest
print('there are ',
      len([ [] for x in enron_data if enron_data[x]['poi'] == True]),
      'persons of interest')

# count number of people whose total payments is not documented
print('there are ',
      len([ [] for x in enron_data if enron_data[x]['total_payments'] == 'NaN']),
          'people whose salary are not recorded')

# find a person's dict key by part of the name
name =  [x for x in enron_data if 'KENNETH'in x]
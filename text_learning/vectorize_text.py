#!/usr/bin/python

import os
import pickle
import re
import sys
import time

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""

from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker

t0 = time.time()
for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    temp_counter = 0
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        temp_counter += 1
        if temp_counter >= 0 :
            path = os.path.join('..', path[:-1])
            # print(path)
            email = open(path, "rb")

            ### use parseOutText to extract the text from the opened email
            words = parseOutText(email)
            ### use str.replace() to remove any instances of the words
            ### ["sara", "shackleton", "chris", "germani"]

            repdict = dict((item, '') for item in ["sara", "shackleton", "chris", "germani", "sshacklensf", "cgermannsf"])
            # repdict = dict((re.escape(k), v) for k, v in repdict.items())
            pattern = re.compile("|".join(repdict.keys()))
            text_string = pattern.sub(lambda m: repdict[re.escape(m.group(0))], words)

            # The following method will work. But note that order in a list is not preserved when converted to set.
            # This method is roughly at the same speed with the regex method above.
            # If using this method, paserOutText should also be modified to output set instead of list.

            # text_string = words - {"sara", "shackleton", "chris", "germani"}

            ### append the text to word_data
            word_data.append(text_string)
            ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris

            from_data.append(0 if name == 'sara'else 1)

            email.close()


t1 = time.time()
print('process took {:0.4f} seconds'.format(t1-t0))
print("emails processed")
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "wb") )
pickle.dump( from_data, open("your_email_authors.pkl", "wb") )



### in Part 4, do TfIdf vectorization here

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
word_data_transformed = vectorizer.fit(word_data)
print(vectorizer.get_feature_names()[34596])
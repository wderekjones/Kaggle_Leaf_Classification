import csv
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, accuracy_score


df = pd.read_csv('train.csv')


#df = df.sort(['species'], ascending = 1)


le = LabelEncoder()


# encode the labels and store in numpy array
labels = le.fit_transform(df['species'])


#filter out the labels and id numbers 
df = df.drop('id', 1)
df = df.drop('species',1)

# transform dataframe to a numpy array
examples = df.as_matrix()


#create the classifier

clf = BaggingClassifier(n_estimators = 1000)

clf.fit(examples,labels)

scores = cross_val_score(clf,examples,labels, cv=5)

preds = clf.predict(examples)

accuracy = accuracy_score(labels,preds)

precision = precision_score(labels,preds,average = 'micro')


print ('cross_val_score: ' + str(scores.mean()))

print ('accuracy: '+ str(accuracy))

print ('precision: '+ str(precision))



test_df = pd.read_csv('test.csv')

test_ids = test_df.id

test_df = test_df.drop('id',1)



test_data = test_df.as_matrix()


preds = clf.predict_proba(test_data)


print preds.shape

print test_ids.shape



#np.insert(test_data,0,preds,axis = 1)








np.savetxt('leaf_preds.csv',preds, delimiter = ',')

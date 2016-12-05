import numpy as np
import pandas as pd


from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import precision_score, accuracy_score



mlp = MLPClassifier(hidden_layer_sizes=(100,),activation='logistic',solver='adam',max_iter=10000,shuffle=True)

clf = BaggingClassifier(base_estimator=mlp,n_estimators=10)

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

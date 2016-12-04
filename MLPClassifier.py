import numpy as np
import pandas as pd


from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, accuracy_score




df = pd.read_csv('train.csv')




le = LabelEncoder()


# encode the labels and store in numpy array
labels = le.fit_transform(df['species'])



#filter out the labels and id numbers
df = df.drop('id', 1)
df = df.drop('species',1)

# transform dataframe to a numpy array
examples = df.as_matrix()


clf = MLPClassifier(hidden_layer_sizes=(100,),activation='logistic',solver='adam',learning_rate='adaptive',alpha=0.0000000000000000000000000000000000001, max_iter=10000)


clf.fit(examples,labels)

preds = clf.predict(examples)

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


np.savetxt('leaf_preds.csv',preds, delimiter = ',')

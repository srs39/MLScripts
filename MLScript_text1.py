import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report


#reading csv files into data frames of profiles and liwc
profiledf = pd.read_csv('/home/samantha/tcss455/training/profile/profile.csv')
liwcdf = pd.read_csv('/home/samantha/tcss455/training/LIWC/LIWC.csv')

#joining the two data frames by user id
textAn = profiledf.join(liwcdf.set_index('userId'), on='userid')
#textAn.describe()
#test data
x = textAn.values[:,  10:]
#print(x.shape)
#age data

y_gender = textAn.values[:, 3]
#print(y_gender.shape)
y_gender = y_gender.astype(int)
X_train, X_test, yGen_train, yGen_test = train_test_split(x,y_gender, test_size = 0.30, random_state=100)


clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=4, min_samples_leaf=5)
clf_gini.fit(X_train, yGen_train)

yGen_pred = clf_gini.predict(X_test)

GenDecisionTreeFile = 'genDTClassifier.pkl'
genDTModel = open(GenDecisionTreeFile, 'wb')
pickle.dump(clf_gini, genDTModel)
genDTModel.close()

#clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
#    max_depth=4, min_samples_leaf=5)
#clf_entropy.fit(X_train, yGen_train)

#y_pred_en = clf_entropy.predict(X_test)

#print ("Gender Accuracy is ", accuracy_score(yGen_test,yGen_pred)*100)

#print ("Gender Accuracy is ", accuracy_score(yGen_test,y_pred_en)*100)

test_age = textAn.copy()
test_age.loc[test_age['age'] < 25, 'age'] = 0
test_age.loc[(test_age['age'] >24) & (test_age['age'] <35), 'age']=1
test_age.loc[(test_age['age'] >34) & (test_age['age'] <50), 'age'] = 2
test_age.loc[test_age['age'] > 49, 'age'] = 3

x_age = test_age.values[:, 10:]
y_age = test_age.values[:, 2]
y_age = y_age.astype(int)

X_train, X_test, yAge_train, yAge_test = train_test_split(x_age,y_age, test_size = 0.30, random_state=100)


clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3)
clf_gini.fit(X_train, yAge_train)

yAge_pred = clf_gini.predict(X_test)

#clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 #   max_depth=3)
#clf_entropy.fit(X_train, yAge_train)

#y_pred_en = clf_entropy.predict(X_test)


ageDecisionTreeFile = 'ageDTClassifier.pkl'
ageDTModel = open(ageDecisionTreeFile, 'wb')
pickle.dump(clf_gini, ageDTModel)
ageDTModel.close()
#print ("Age Accuracy is ", accuracy_score(yAge_test,yAge_pred)*100)
#print("Age Accuracy is: ", accuracy_score(yAge_test, y_pred_en)*100)


Y_open = textAn.values[ :, 4]
X_train, X_test, yO_train, yO_test = train_test_split(x,Y_open, test_size = 0.30, random_state=100)
openTree = DecisionTreeRegressor(max_depth = 3)
openTree.fit(X_train, yO_train)
yO_pred = openTree.predict(X_test)

opeDecisionTreeFile = 'opeDTClassifier.pkl'
opeDTModel = open(opeDecisionTreeFile, 'wb')
pickle.dump(openTree, opeDTModel)
opeDTModel.close()
#print("open Accuracy is", r2_score(yO_test, yO_pred, multioutput='variance_weighted'))

y_con = textAn.values[ : , 5]

X_train, X_test, yC_train, yC_test = train_test_split(x,y_con, test_size = 0.30, random_state=100)
conTree = DecisionTreeRegressor(max_depth = 3)
conTree.fit(X_train, yC_train)
yC_pred = conTree.predict(X_test)

conDecisionTreeFile = 'conDTClassifier.pkl'
conDTModel = open(conDecisionTreeFile, 'wb')
pickle.dump(conTree, conDTModel)
conDTModel.close()
#print("con Accuracy is", r2_score(yC_test, yC_pred, multioutput='variance_weighted'))

y_ext = textAn.values[ : , 6]

X_train, X_test, yE_train, yE_test = train_test_split(x,y_ext, test_size = 0.30, random_state=100)
extTree = DecisionTreeRegressor(max_depth = 3)
extTree.fit(X_train, yE_train)
yE_pred = extTree.predict(X_test)

extDecisionTreeFile = 'extDTClassifier.pkl'
extDTModel = open(extDecisionTreeFile, 'wb')
pickle.dump(extTree, extDTModel)
extDTModel.close()
#print("Ext Accuracy is", r2_score(yE_test, yE_pred, multioutput='variance_weighted'))

y_agr = textAn.values[ : , 7]

X_train, X_test, yA_train, yA_test = train_test_split(x,y_agr, test_size = 0.30, random_state=100)
agrTree = DecisionTreeRegressor(max_depth = 3)
agrTree.fit(X_train, yA_train)
yA_pred = agrTree.predict(X_test)

agrDecisionTreeFile = 'agrDTClassifier.pkl'
agrDTModel = open(agrDecisionTreeFile, 'wb')
pickle.dump(agrTree, agrDTModel)
agrDTModel.close()
#print("con Accuracy is", r2_score(yA_test, yA_pred, multioutput='variance_weighted'))

y_neu = textAn.values[ : , 8]

X_train, X_test, yN_train, yN_test = train_test_split(x,y_neu, test_size = 0.30, random_state=100)
neuTree = DecisionTreeRegressor(max_depth = 3)
neuTree.fit(X_train, yN_train)
yN_pred = neuTree.predict(X_test)

neuDecisionTreeFile = 'neuDTClassifier.pkl'
neuDTModel = open(neuDecisionTreeFile, 'wb')
pickle.dump(neuTree, neuDTModel)
neuDTModel.close()
#print("Neu Accuracy is", r2_score(yN_test, yN_pred, multioutput='variance_weighted'))

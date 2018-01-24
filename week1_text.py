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
import io, sys, os, csv
import xml.etree.ElementTree as et
import logging
#from dicttoxml import dicttoxml

input_dir = str(sys.argv[1])
output_dir = str(sys.argv[2])

profile_csv = "{}/{}".format(input_dir, "profile/profile.csv")
profile_df = pd.read_csv(profile_csv)

liwc_csv = "{}/{}".format(input_dir, "LIWC/LIWC.csv")
liwc_df = pd.read_csv(liwc_csv)

x = liwc_df.values[:, 2: ]
genDT_pkl = open('genDTClassifier.pkl', 'rb')
genDT_model = pickle.load(genDT_pkl)

gen_pred = genDT_model.predict(x)
genDT_pkl.close()
#print("genPkl")
ageDT_pkl = open('ageDTClassifier.pkl', 'rb')
ageDT_model = pickle.load(ageDT_pkl)

age_pred = ageDT_model.predict(x)
ageDT_pkl.close()
#print("agePkl")
openDT_pkl = open('opeDTClassifier.pkl', 'rb')
openDT_model = pickle.load(openDT_pkl)

open_pred = openDT_model.predict(x)
openDT_pkl.close()
conDT_pkl = open('conDTClassifier.pkl','rb')
conDT_model = pickle.load(conDT_pkl)
con_pred = conDT_model.predict(x)
conDT_pkl.close()

#print('openpkl')
extDT_pkl = open('extDTClassifier.pkl', 'rb')
extDT_model = pickle.load(extDT_pkl)

ext_pred = extDT_model.predict(x)
extDT_pkl.close()
print('extPkl')
agrDT_pkl = open('agrDTClassifier.pkl', 'rb')
agrDT_model = pickle.load(agrDT_pkl)

agr_pred = agrDT_model.predict(x)
agrDT_pkl.close()
print('agrpkl')
neuDT_pkl = open('neuDTClassifier.pkl', 'rb')
neuDT_model = pickle.load(neuDT_pkl)

neu_pred = neuDT_model.predict(x)
neuDT_pkl.close()
#print('neupkl')

se = pd.Series(age_pred)
profile_df['age'] = se.values
profile_df.loc[profile_df['age'] == 0, 'age'] = 'xx-24'
profile_df.loc[profile_df['age'] == 1, 'age'] = '25-34'
profile_df.loc[profile_df['age'] == 2, 'age'] = '35-49'
profile_df.loc[profile_df['age'] == 3, 'age'] = '50-xx'
se = pd.Series(gen_pred)
profile_df['gender'] = se.values
profile_df.loc[profile_df['gender'] == 0, 'gender'] = 'male'
profile_df.loc[profile_df['gender'] == 1, 'gender'] = 'female'
se = pd.Series(open_pred)
profile_df['ope'] = se.values
se = pd.Series(con_pred)
profile_df['con'] = se.values
se = pd.Series(ext_pred)
profile_df['ext'] = se.values
se = pd.Series(agr_pred)
profile_df['agr'] = se.values
se = pd.Series(neu_pred)
profile_df['neu'] = se.values
for i in range(0, profile_df.shape[0]):
    elem = et.Element("user", attrib={
        "id": profile_df.loc[i,'userid'],
        "age_group": profile_df.loc[i,'age'],
        "gender": profile_df.loc[i,'gender'],
       "extrovert": '{:.2f}'.format(profile_df.loc[i,'ext']),
        "neurotic": '{:.2f}'.format(profile_df.loc[i,'neu']),
        "agreeable": '{:.2f}'.format(profile_df.loc[i,'agr']),
        "conscientious": '{:.2f}'.format(profile_df.loc[i,'con']),
        "open": '{:.2f}'.format(profile_df.loc[i,'ope'])})
    tree = et.ElementTree(element=elem)
    filename = "{}/{}.xml".format(output_dir, profile_df.loc[i, 'userid'])
    with open(filename, 'wb') as f:
        tree.write(f)

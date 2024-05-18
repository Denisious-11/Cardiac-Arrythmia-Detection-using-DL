###import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
import operator
import pickle
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint 



###load dataset
data=pd.read_csv("Project_Dataset/data_arrhythmia.csv")
# print(data.head())
# print(data.columns)

data.columns=data.columns.str.replace(';',',')
new_data = data.apply(lambda x: x.replace({';':','}, regex=True))
# print(new_data.head())
# print(new_data.columns)

new_data.to_csv('Project_Dataset/preprocessed.csv')

#####################################################################################################

#load preprocessed dataset
data=pd.read_csv("Project_Dataset/preprocessed.csv")
print(data.head())
print(data.columns)


## Count the Number of Occurrences of 'diagnosis'
print(data['diagnosis'].value_counts())

#print columns with missing values and its occurences
#print(data.isnull().sum())

from preprocess import *

#function call
non_features=get_preprocess1(data)

print(non_features)
print(len(non_features))

new_df = data.copy()
# delete non-informative features
new_df = data.drop(columns=non_features) 

print(new_df.columns)

#'J' column has many missing values
new_df1 = new_df.copy()
new_df1 = new_df1.drop(columns=['J'])
print(new_df1.columns)

# apply the dtype attribute
result = new_df1.dtypes
print(result)


new_df1 = new_df1.replace('?',0)

# num_cols=new_df1._get_numeric_data()
# num_cols[num_cols<0]=0
# print(num_cols)

#data division (independent variable & dependent variable)
y=new_df1['diagnosis']
x=new_df1.drop(['diagnosis'],axis=1)


# feature selection(function call)
fs = select_features(x, y)


column_names=[]
# iterating the columns
j=0
for col in x.columns:
    print(col)
    print(j)
    column_names.append(col)
    j=j+1

print(column_names)
print(len(column_names))

feature_list=[]
# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
	feature_list.append(fs.scores_[i])
# # plot the scores
# pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
# pyplot.show()

print(feature_list)
print(len(feature_list))

dictionary = dict(zip(column_names, feature_list))
sorted_d = dict( sorted(dictionary.items(), key=operator.itemgetter(1),reverse=True))

#Creating new dataframe based on feature scores (15 features)
my_data=new_df1[['age','sex','height','weight','qrs_duration','p-r_interval','q-t_interval','t_interval','p_interval','qrs','QRST','heart_rate','q_wave','r_wave','s_wave','diagnosis']]
print(my_data)
print(my_data.columns)

#removing "others" and low category rows
my_data.drop(my_data.index[my_data['diagnosis'] == 16], inplace=True)
my_data.drop(my_data.index[my_data['diagnosis'] == 3], inplace=True)
my_data.drop(my_data.index[my_data['diagnosis'] == 4], inplace=True)
my_data.drop(my_data.index[my_data['diagnosis'] == 5], inplace=True)
my_data.drop(my_data.index[my_data['diagnosis'] == 7], inplace=True)
my_data.drop(my_data.index[my_data['diagnosis'] == 8], inplace=True)
my_data.drop(my_data.index[my_data['diagnosis'] == 9], inplace=True)
my_data.drop(my_data.index[my_data['diagnosis'] == 14], inplace=True)
my_data.drop(my_data.index[my_data['diagnosis'] == 15], inplace=True)


my_data['diagnosis']=my_data['diagnosis'].replace(1,0)
my_data['diagnosis']=my_data['diagnosis'].replace(2,1)
my_data['diagnosis']=my_data['diagnosis'].replace(6,2)
my_data['diagnosis']=my_data['diagnosis'].replace(10,3)

#my_data.to_csv("Project_Dataset/final_dataset_preprocessed.csv")



print(my_data)

#data division
y_final = my_data['diagnosis']
x_final = my_data.drop(['diagnosis'], axis=1)




# TRAIN - TEST SPLITTING
x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, test_size=0.2)
print("\nTraining set")
print(x_train.shape)
print(y_train.shape)
print("\nTesting set")
print(x_test.shape)
print(y_test.shape)

print(x_train)


#Data balancing using SMOTE
counter = Counter(y_train)
print("__________________BEFORE::::::", counter)

smt = SMOTE(k_neighbors=1)

x_train_sm, y_train_sm = smt.fit_resample(x_train, y_train)

counter = Counter(y_train_sm)
print("___________________AFTER:::::::", counter)


print("x_train_sm shape:", x_train_sm.shape)
print("y_train_sm shape:", y_train_sm.shape)

#perform label encoding
# encoder = OneHotEncoder()
# y_train_sm = encoder.fit_transform(np.array(y_train_sm).reshape(-1,1)).toarray()

#Perform standardization
scaler = StandardScaler()
x_train_sm = scaler.fit_transform(x_train_sm)
x_test = scaler.transform(x_test)
print(x_train_sm.shape, y_train_sm.shape, x_test.shape, y_test.shape)
pickle.dump(scaler,open('Project_Extra/scaler.pkl','wb'))

#Applying dimension expansion
x_train_sm = np.expand_dims(x_train_sm, axis=2)
x_test = np.expand_dims(x_test, axis=2)
print(x_train_sm.shape)
print(y_train_sm.shape)
print(x_test.shape)
print(y_test.shape)


#model loading
from model import model_customized

model=model_customized(x_train_sm)


#learning rate reduce , if quantity stops improving
#rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=1, patience=2, min_lr=0.0000001)

#saving the model(checkpoint)
checkpoint=ModelCheckpoint("Project_Saved_Models/trained_model.h5",monitor="accuracy",save_best_only=True,verbose=1)#when training deep learning model,checkpoint is "WEIGHT OF THE MODEL"
#Training
history=model.fit(x_train_sm, y_train_sm, batch_size=16, epochs=50, validation_data=(x_test, y_test), callbacks=[checkpoint])



#plot accuracy and loss 
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()






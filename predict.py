#import necessary libraries
import pandas as pd
import joblib
import numpy as np
import pickle
from tensorflow.keras.models import load_model



#loading the saved model
loaded_model=load_model("Project_Saved_Models/trained_model.h5")
#loading standardscaler
scaler=pickle.load(open('Project_Extra/scaler.pkl','rb'))

info=[]
parameters=['age','sex','height','weight','qrs_duration','p-r_interval','q-t_interval','t_interval','p_interval','qrs','QRST','heart_rate','q_wave','r_wave','s_wave']


age=input("age : ")
info.append(age)
sex=input("sex : ")
info.append(sex)
height=input("height : ")
info.append(height)
weight=input("weight : ")
info.append(weight)
qrs_duration=input("qrs_duration : ")
info.append(qrs_duration)
p_r_interval=input("p-r_interval : ")
info.append(p_r_interval)
q_t_interval=input("q-t_interval : ")
info.append(q_t_interval)
t_interval=input("t_interval : ")
info.append(t_interval)
p_interval=input("p_interval : ")
info.append(p_interval)
qrs=input("qrs : ")
info.append(qrs)
QRST=input("QRST : ")
info.append(QRST)
heart_rate=input("heart_rate : ")
info.append(heart_rate)
q_wave=input("q_wave : ")
info.append(q_wave)
r_wave=input("r_wave : ")
info.append(r_wave)
s_wave=input("s_wave : ")
info.append(s_wave)


my_dict=dict(zip(parameters,info))

#convert dict into dataframe
my_data=pd.DataFrame(my_dict,index=[0])

feat=np.array(my_data)
# print(feat)
#perform standardization
feat=scaler.transform(feat)
# print(feat)
#expand dimension
feat = np.expand_dims(feat, axis=2)
# print(feat)
# print(feat.shape)

# dataframe is putting into the MODEL to make PREDICTION
my_pred = loaded_model.predict(feat)
print(my_pred)
my_pred=np.argmax(my_pred)
print(my_pred)

print("\n*************Result**************")

if my_pred==0:
	print("[INFO] : Safe ")

if my_pred==1:
	print("Cardiac Arrythmia Detected")
	print("\nType")
	print("Ischemic changes (Coronary Artery Disease)")

if my_pred==2:
	print("Cardiac Arrythmia Detected")
	print("\nType")
	print("Sinus bradycardy")

if my_pred==3:
	print("Cardiac Arrythmia Detected")
	print("\nType")
	print("Right bundle branch block")


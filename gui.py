from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
import cv2 as cv
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tkinter import messagebox

a = Tk()
a.title("Cardiac Arrythmia Detection")
a.geometry("1350x670")
a.minsize(1350,670)
a.maxsize(1350,670)


#loading the saved model
loaded_model=load_model("Project_Saved_Models/trained_model.h5")
#loading standardscaler
scaler=pickle.load(open('Project_Extra/scaler.pkl','rb'))

def prediction(e_var0,e_var1,e_var2,e_var3,e_var4,e_var5,e_var6,e_var7,e_var8,e_var9,e_var10,e_var11,e_var12,e_var13,e_var14):
    age=e_var0.get()
    sex=e_var1.get()
    height=e_var2.get()
    weight=e_var3.get()
    qrs_duration=e_var4.get()
    p_r_interval=e_var5.get()
    q_t_interval=e_var6.get() 
    t_interval=e_var7.get() 
    p_interval=e_var8.get() 
    qrs=e_var9.get() 
    QRST=e_var10.get() 
    heart_rate=e_var11.get() 
    q_wave=e_var12.get() 
    r_wave=e_var13.get() 
    s_wave=e_var14.get() 

    if((age=='') or (sex=='') or (height=='') or (weight=='') or (qrs_duration=='') or (p_r_interval=='') or (q_t_interval=='') or (t_interval=='') or (p_interval=='') or (qrs=='') or (QRST=='') or (heart_rate=='') or (q_wave=='') or (r_wave=='') or (s_wave=='')):
        messagebox.showinfo("Warning", "Please Fill all fields")                                                                                                                                                                                                                                                                                                            
    else:

        list_box.insert(1, "Collect Values")
        list_box.insert(2, "")
        list_box.insert(3, "Loading Trained Model")
        list_box.insert(4, "")
        list_box.insert(5, "Loading Standardscaler")
        list_box.insert(6, "")
        list_box.insert(7, "Preprocessing")
        list_box.insert(8, "")
        list_box.insert(9, "Prediction")

       

        info=[]
        parameters=['age','sex','height','weight','qrs_duration','p-r_interval','q-t_interval','t_interval','p_interval','qrs','QRST','heart_rate','q_wave','r_wave','s_wave']

        info.append(age)
        info.append(sex)
        info.append(height)
        info.append(weight)
        info.append(qrs_duration)
        info.append(p_r_interval)
        info.append(q_t_interval)
        info.append(t_interval)
        info.append(p_interval)
        info.append(qrs)
        info.append(QRST)
        info.append(heart_rate)
        info.append(q_wave)
        info.append(r_wave)
        info.append(s_wave)


        my_dict=dict(zip(parameters,info))

        #convert dict into dataframe
        my_data=pd.DataFrame(my_dict,index=[0])

        feat=np.array(my_data)
        #perform standardization
        feat=scaler.transform(feat)
        #expand dimension
        feat = np.expand_dims(feat, axis=2)

        # dataframe is putting into the MODEL to make PREDICTION
        my_pred = loaded_model.predict(feat)
        my_pred=np.argmax(my_pred)
        print(my_pred)

        print("\n*************Result**************")

        if my_pred==0:
            print("[INFO] : Safe ")
            final_out1=" "
            final_out="[INFO] : Safe "

        if my_pred==1:
            print("Cardiac Arrythmia Detected")
            print("\nType")
            print("Ischemic changes (Coronary Artery Disease)")
            final_out1="Cardiac Arrythmia Detected"
            final_out="Ischemic changes (Coronary Artery Disease)"

        if my_pred==2:
            print("Cardiac Arrythmia Detected")
            print("\nType")
            print("Sinus bradycardy")
            final_out1="Cardiac Arrythmia Detected"
            final_out="Sinus bradycardy"


        if my_pred==3:
            print("Cardiac Arrythmia Detected")
            print("\nType")
            print("Right bundle branch block")
            final_out1="Cardiac Arrythmia Detected"
            final_out="Right bundle branch block"


        out_label.config(text=final_out)
        out_label1.config(text=final_out1)

 
def Check():
    global f
    f.pack_forget()

    f = Frame(a, bg="white")
    f.pack(side="top", fill="both", expand=True)

    global f1
    f1 = Frame(f, bg="LightSkyBlue1")
    f1.place(x=0, y=0, width=900, height=690)
    f1.config()

    input_label = Label(f1, text="INPUT", font="arial 16 bold", bg="LightSkyBlue1")
    input_label.pack(padx=0, pady=10)

    label1 = Label(f1, text="Age :", font="arial 12 bold", bg="LightSkyBlue1")
    label1.place(x=40, y=70)
    label2 = Label(f1, text="Gender :",
                   font="arial 12 bold", bg="LightSkyBlue1")
    label2.place(x=40, y=120)
    label3 = Label(f1, text="Height :",
                   font="arial 12 bold", bg="LightSkyBlue1")
    label3.place(x=40, y=170)
    label4 = Label(f1, text="Weight :",
                   font="arial 12 bold", bg="LightSkyBlue1")
    label4.place(x=40, y=220)
    label5 = Label(f1, text="qrs_duration :",
                   font="arial 12 bold", bg="LightSkyBlue1")
    label5.place(x=40, y=270)
    label6 = Label(f1, text="p-r_interval :",
                   font="arial 12 bold", bg="LightSkyBlue1")
    label6.place(x=40, y=320)
    label7 = Label(f1, text="q-t_interval :",
                   font="arial 12 bold", bg="LightSkyBlue1")
    label7.place(x=40, y=370)
    label8 = Label(f1, text="t_interval :",
                   font="arial 12 bold", bg="LightSkyBlue1")
    label8.place(x=40, y=420)

    label9 = Label(f1, text="p_interval :",
                   font="arial 12 bold", bg="LightSkyBlue1")
    label9.place(x=440, y=70)
    label10 = Label(f1, text="qrs :",
                   font="arial 12 bold", bg="LightSkyBlue1")
    label10.place(x=440, y=120)
    label11 = Label(f1, text="QRST :", font="arial 12 bold", bg="LightSkyBlue1")
    label11.place(x=440, y=170)
    label12 = Label(f1, text="Heart_rate :",
                   font="arial 12 bold", bg="LightSkyBlue1")
    label12.place(x=440, y=220)
    label13 = Label(f1, text="Q_wave :",
                   font="arial 12 bold", bg="LightSkyBlue1")
    label13.place(x=440, y=270)
    label14 = Label(f1, text="R_wave :",
                   font="arial 12 bold", bg="LightSkyBlue1")
    label14.place(x=440, y=320)
    label15 = Label(f1, text="S_wave :",
                   font="arial 12 bold", bg="LightSkyBlue1")
    label15.place(x=440, y=370)

    global e_var0,e_var1,e_var2,e_var3,e_var4,e_var5,e_var6,e_var7,e_var8,e_var9
    global e_var10,e_var11,e_var12,e_var13,e_var14
    e_var0=StringVar()
    e_var1=StringVar()
    e_var2=StringVar()
    e_var3=StringVar()
    e_var4=StringVar()
    e_var5=StringVar()
    e_var6=StringVar()
    e_var7=StringVar()
    e_var8=StringVar()
    e_var9=StringVar()
    e_var10=StringVar()
    e_var11=StringVar()
    e_var12=StringVar()
    e_var13=StringVar()
    e_var14=StringVar()
    message1=StringVar()


    entry0 = Entry(f1, textvariable=e_var0, bd=2, width=25)
    entry0.place(x=210, y=70)
    entry1 = Entry(f1, textvariable=e_var1, bd=2, width=25)
    entry1.place(x=210, y=120)
    entry2 = Entry(f1, textvariable=e_var2, bd=2, width=25)
    entry2.place(x=210, y=170)
    entry3 = Entry(f1, textvariable=e_var3, bd=2, width=25)
    entry3.place(x=210, y=220)
    entry4 = Entry(f1, textvariable=e_var4, bd=2, width=25)
    entry4.place(x=210, y=270)
    entry5 = Entry(f1, textvariable=e_var5, bd=2, width=25)
    entry5.place(x=210, y=320)
    entry6 = Entry(f1, textvariable=e_var6, bd=2, width=25)
    entry6.place(x=210, y=370)
    entry7 = Entry(f1, textvariable=e_var7, bd=2, width=25)
    entry7.place(x=210, y=420)

    entry8 = Entry(f1, textvariable=e_var8, bd=2, width=25)
    entry8.place(x=640, y=70)
    entry9 = Entry(f1, textvariable=e_var9, bd=2, width=25)
    entry9.place(x=640, y=120)
    entry10 = Entry(f1, textvariable=e_var10, bd=2, width=25)
    entry10.place(x=640, y=170)
    entry11 = Entry(f1, textvariable=e_var11, bd=2, width=25)
    entry11.place(x=640, y=220)
    entry12 = Entry(f1, textvariable=e_var12, bd=2, width=25)
    entry12.place(x=640, y=270)
    entry13 = Entry(f1, textvariable=e_var13, bd=2, width=25)
    entry13.place(x=640, y=320)
    entry14 = Entry(f1, textvariable=e_var14, bd=2, width=25)
    entry14.place(x=640, y=370)

    predict_button = Button(
        f1, text="Predict",width=20,height=2, command=lambda: prediction(e_var0,e_var1,e_var2,e_var3,e_var4,e_var5,e_var6,e_var7,e_var8,e_var9,e_var10,e_var11,e_var12,e_var13,e_var14), bg="hot pink")
    predict_button.pack(side="bottom", pady=150)
    global f2
    f2 = Frame(f, bg="turquoise")
    f2.place(x=900, y=320, width=470, height=370)
    f2.config(pady=20)

    result_label = Label(f2, text="RESULT", font="arial 16 bold", bg="turquoise")
    result_label.pack(padx=0, pady=0)

    global out_label1
    out_label1 = Label(f2, text="", bg="turquoise", font="arial 16 bold")
    out_label1.pack(pady=40)

    global out_label
    out_label = Label(f2, text="", bg="turquoise", font="arial 16 bold")
    out_label.pack(pady=20)

    f3 = Frame(f, bg="darksalmon")
    f3.place(x=900, y=0, width=470, height=320)
    f3.config()

    name_label = Label(f3, text="PROCESS", font="arial 16 bold", bg="darksalmon")
    name_label.pack(pady=20)

    global list_box
    list_box = Listbox(f3, height=12, width=31)
    list_box.pack()


def Home():
    global f
    f.pack_forget()

    f = Frame(a, bg="salmon")
    f.pack(side="top", fill="both", expand=True)

    front_image = Image.open("Project_Extra/home1.jpg")
    front_photo = ImageTk.PhotoImage(front_image.resize((a.winfo_width(), a.winfo_height()), Image.ANTIALIAS))
    front_label = Label(f, image=front_photo)
    front_label.image = front_photo
    front_label.pack()

    heading_label = Label(f, text="Cardiac Arrythmia Detection", font="arial 26 bold", bg="white")
    heading_label.place(x=500, y=50)

  

f = Frame(a, bg="salmon")
f.pack(side="top", fill="both", expand=True)
front_image1 = Image.open("Project_Extra/home1.jpg")
front_photo1 = ImageTk.PhotoImage(front_image1.resize((1350,670), Image.ANTIALIAS))
front_label1 = Label(f, image=front_photo1)
front_label1.image = front_photo1
front_label1.pack()

heading_label = Label(f, text="Cardiac Arrythmia Detection", font="arial 26 bold", bg="white")
heading_label.place(x=500, y=50)

m = Menu(a)
m.add_command(label="Home", command=Home)
checkmenu = Menu(m)
m.add_command(label="Check", command=Check)
a.config(menu=m)


a.mainloop()

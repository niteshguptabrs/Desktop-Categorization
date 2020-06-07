#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Activation,BatchNormalization,Dropout
import cv2
# load the model
# load an image from file
import tkinter
from tkinter import *
from tkinter.filedialog import askdirectory
from tkinter import messagebox
from PIL import Image, ImageTk
root=Tk()
root.geometry('900x487')
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH,expand=1)
root.title('ICRR')
frame.config(background='light blue')
label = Label(frame, text="Image Categorization and Redundancy Remover",bg='light blue',font=('Times 20 bold'))
label.pack(side=TOP)
image=Image.open("back.png")
filename = ImageTk.PhotoImage(image,master=frame)
background_label = Label(frame,image=filename)
background_label.pack(side=TOP)

f1=""
def mfileopen():
    global f1
    f1=askdirectory()
    mess=messagebox.askyesno(title='Folder Selected',message='Do you wish to work with selected folder ?')
    if mess==1:
        label=Label(frame,text="Folder selected: "+f1,bg='yellow',font=('Times 15 bold'))
        label.place(x=105,y=350)
        
import glob
import numpy as np
from keras.applications.vgg16 import preprocess_input
def extract_vector(path,model):
    VGG_featlist=[]
    imname=[]
    for im in glob.glob(path):
        imname.append(im)
        im = cv2.imread(im)
        im = cv2.resize(im,(224,224))
        img = preprocess_input(np.expand_dims(im.copy(), axis=0))
        vgg_feature=model.predict(img)
        vgg_feature_np = np.array(vgg_feature)
        VGG_featlist.append(vgg_feature_np.flatten())
    return np.array(VGG_featlist),imname

def cap():
    model=Sequential()
    model.add(VGG16(include_top=False, pooling='avg', weights="imagenet"))
    model.layers[0].trainable = False
    path=f1+"/*.jpg"
    array,imname=extract_vector(path,model)
    # In[7]:
    from sklearn.cluster import KMeans
    kmeans=KMeans(n_clusters=4,random_state=0).fit(array)
    X=kmeans.labels_
    r=np.where(X==0)[0]
    s=np.where(X==1)[0]
    t=np.where(X==2)[0]
    u=np.where(X==3)[0]
    #v=np.where(X==4)[0]
    rn=[]
    for i in r:
        rn.append(imname[i])

    sn=[]
    for i in s:
        sn.append(imname[i])

    tn=[]
    for i in t:
        tn.append(imname[i])

    un=[]
    for i in u:
        un.append(imname[i])

    #vn=[]
    #for i in v:
    #    vn.append(imname[i])

    import shutil, os
    import cv2
    import os
    dir = os.path.join(f1+"/f1")
    if not os.path.exists(dir):
        os.mkdir(dir)
    for f in rn:
        shutil.move(f,f1+'/f1')

    dir = os.path.join(f1+"/f2")
    if not os.path.exists(dir):
        os.mkdir(dir)
    for f in sn:
        shutil.move(f,f1+'/f2')

    dir = os.path.join(f1+"/f3")
    if not os.path.exists(dir):
        os.mkdir(dir)
    for f in tn:
        shutil.move(f,f1+'/f3')

    dir = os.path.join(f1+"/f4")
    if not os.path.exists(dir):
        os.mkdir(dir)
    for f in un:
        shutil.move(f,f1+'/f4')

    #dir = os.path.join(f1+"/f5")
    #if not os.path.exists(dir):
    #    os.mkdir(dir)
    #for f in vn:
    #    shutil.move(f,f1+'/f5')
    
    mess=messagebox.askyesno(title='Success',message='Task completed!! Do you want to quit ?')
    if mess==1:
        root.destroy()#close the window

def mquit():
    mess=messagebox.askyesno(title='quit',message='Are you sure to quit')
    if mess==1:
        root.destroy()#close the window
button1=tkinter.Button(frame,text="Select Target Folder",padx=5,pady=5,width=39,bg='yellow',fg='black',relief=GROOVE,command=mfileopen,font=('helvetica 15 bold'))
button1.place(x=105,y=104)       
button2=tkinter.Button(frame,text="Run Model",padx=5,pady=5,width=39,bg='yellow',fg='black',relief=GROOVE,command=cap,font=('helvetica 15 bold'))
button2.place(x=105,y=176)
button3=tkinter.Button(frame,text="Quit",padx=5,pady=5,width=39,bg='yellow',fg='black',relief=GROOVE,command=mquit,font=('helvetica 15 bold'))       
button3.place(x=105,y=250)
    
root.mainloop()


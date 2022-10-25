
from tkinter import *
from tkinter import filedialog
from turtle import color
from PIL import ImageTk,Image
import time
import tensorflow as tf
from keras.models import load_model 
import numpy as np 

root=Tk()
root.title('OCEAN DEBRIS DETECTOR App')
root.iconbitmap('')
mylabel=Label(root,text='Detecting ocean debris using satellite data')
mylabel.grid(row=0,column=0,columnspan=7)


background_img=ImageTk.PhotoImage(Image.open('./ocean_challenge_code/deb.jpg'))
mylabel_2=Label(image=background_img)
mylabel_2.grid(row=1,column=1,columnspan=10)

model=load_model('ocean.h5')
global open_file


def upload():
    open_file = filedialog.askopenfilename(filetypes=[('JPG files','*.jpg'),('PNG Files','*.png')]) # Returns opened path as str 
    time.sleep(10)
    global top
    top=Toplevel(root)
    top.title('UPLOAD')
    top.geometry("200x100")
    sus=Label(top,text='FILE UPLOADED SUCCESSFULLY')
    sus.grid(row=0,column=0)
    top.after(4000,lambda:top.destroy())
    return open_file
Upload_Xray_button=Button(root,text='UPLOAD',command=upload)
Upload_Xray_button.grid(row=2,column=2)


def predict_debris():
    top1=Toplevel(root)
    top1.title('UPLOAD')
    top1.geometry("100x100")
    
    ocean_image=open_file
    img = tf.keras.utils.load_img(ocean_image, target_size=(256, 256))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_names=["debris","no_debris"]
    top1=Label(text="This image is most likely an xray with  {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))
    top1.grid(row=1,column=0)
    return
Predict_disease_button=Button(root,text='PREDICTION',background='yellow',command=predict_debris)   
Predict_disease_button.grid(row=2,column=4) 



def close():
    return
Close_button=Button(root,text='CLOSE_APP',background='red',foreground='blue',command=quit)   
Close_button.grid(row=2,column=6) 

root.mainloop()
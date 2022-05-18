import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter.filedialog import askopenfilename,asksaveasfilename
from PIL import Image, ImageTk, ImageFilter, ImageEnhance, ImageOps
import os
import function
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg ,NavigationToolbar2Tk

def GrayImage():
    global done,last_label
    done = function.GrayImage(image_path)
    last_label ="Gray Image"
    var2.set(last_label)
    SecondImage()

def add():
    global done,last_label
    Input = int(entry2.get())
    done = function.add(image_path,Input)
    last_label ="Add {}".format(Input)
    var2.set(last_label)
    SecondImage()

def contrast():
    global done,last_label
    done = function.contrast(image_path)
    last_label ="Contrast Stretching"
    var2.set(last_label)
    SecondImage()

def GSApproch1():
    global done,last_label
    done = function.GSApproch1(image_path)
    last_label ="GrayScale Approch1"
    var2.set(last_label)
    SecondImage()
    
def GSApproch2():
    global done
    done = function.GSApproch2(image_path)
    last_label ="GrayScale Approch2"
    var2.set(last_label)
    SecondImage()    

def GrayScaling():
    global done,last_label
    scale = int(scale_n.get())
    done = function.GrayScaling(image_path,scale)
    last_label ="Gray Scale {}".format(scale)
    var2.set(last_label)
    SecondImage()
    
def Negative():
    global done,last_label
    done = function.Negative(image_path)
    last_label ="Negative Image"
    var2.set(last_label)
    SecondImage()   
    
def log():
    global done,last_label
    done = function.log(image_path)
    last_label ="Log Image"
    var2.set(last_label)
    SecondImage()  
 
def medianFilter():
    global done,last_label
    done = function.medianFilter(image_path)
    last_label ="Median Filter"
    var2.set(last_label)
    SecondImage()  

def power():
    global done,last_label
    gamma = float(entry3.get())
    done = function.power(image_path,gamma)
    last_label ="Power {} Image".format(gamma)
    var2.set(last_label)
    SecondImage()      

def Avarage():
    global done,last_label
    done = function.Avarage(image_path)
    last_label ="Avarage"
    var2.set(last_label)
    SecondImage()  
    
def sharp():
    global done,last_label
    done = function.sharp(image_path)
    last_label ="Sharping"
    var2.set(last_label)
    SecondImage()    
    
def clearImage1():
    canvas2.delete("all")    
    var1.set("Origional Image")
    
def clearImage2():
    canvas3.delete("all")    
    var2.set("Second Image")
    
def selected():
    global image_path, image, orignal_size, entry,origional_length
    clearImage1() 
    image_path = filedialog.askopenfilename(initialdir=os.getcwd()) 
    image = Image.open(image_path)
    origional_length = len(image_path)
    image.thumbnail((380, 380))
    image1 = ImageTk.PhotoImage(image)
    orignal_size = (image1.width(), image1.height())
    canvas2.create_image(200, 200, image=image1)
    canvas2.image=image1  
    Directory.set(image_path)   
    var1.set("Origional Image")        
    clearImage2()
      
def SecondImage():  
    global image_path, done,resized,orignal_size,img
    resized = cv.resize(done, orignal_size)   
    img = ImageTk.PhotoImage(Image.fromarray(resized))
    canvas3.create_image(200, 200, image=img)
    canvas3.image=img     
    
def reuse():
    global image_path, done,resized,orignal_size,img,last_label 
    resized = cv.resize(done, orignal_size)   
    img = ImageTk.PhotoImage(Image.fromarray(resized))
    canvas2.create_image(200, 200, image=img)
    canvas2.image=img
    var1.set(last_label)
    imgg = Image.fromarray(done)
    file_name, file_ext = os.path.splitext(image_path)
    file_name = file_name + "({})".format(last_label)
    new_image_path = r'{}.{}'.format(file_name,file_ext)      
    imgg.save(new_image_path)   
    currant_length =  len(image_path)
    new_length = len(new_image_path)
    if (origional_length < currant_length and currant_length<new_length):
        os.remove(image_path) 
    image_path = new_image_path    
    clearImage2()
                          
def save():
    imgg = Image.fromarray(done)
    ext = image_path.split(".")[-1]
    file=asksaveasfilename(defaultextension =f".{ext}",filetypes=[("All Files","*.*"),("PNG file","*.png"),("jpg file","*.jpg")])
    if file: 
        imgg.save(file)                          
        
ws = Tk()
ws.title("Image processing")
ws.geometry("840x750")
ws.configure(bg='#85929E')  
# ws.iconbitmap('python.png')
                           
button1 = Button(ws, text="Select Image", bg='#52BE80', fg='black', font=('ariel 15 bold'),cursor="hand2" ,relief=GROOVE, command=selected)
button1.place(x=60, y=700)
button2 = Button(ws, text="Save", width=12, bg='#D6EAF8', fg='black', font=('ariel 15 bold'),cursor="hand2", relief=GROOVE, command=save)
button2.place(x=440, y=700)
button3 = Button(ws, text="Exit", width=12, bg='#CD6155', fg='black', font=('ariel 15 bold'),cursor="hand2", relief=GROOVE, command=ws.destroy)
button3.place(x=660, y=700)
button4 = Button(ws, text="Reuse", width=12, bg='#afafaf', fg='black', font=('ariel 15 bold'),cursor="hand2", relief=GROOVE, command=reuse)
button4.place(x=250, y=700)

Button( ws,text='Gray Scaling',width=12,font=("ariel 17 bold"),bg='#3498DB',cursor="hand2",command= GrayScaling).place(x=30, y=12)
Button( ws,text='ContrastStretch',width=12,font=("ariel 17 bold"),bg='#3498DB',cursor="hand2",command= contrast).place(x=30, y=70)
Button( ws,text='Negative Image',width=12,font=("ariel 17 bold"),bg='#3498DB',cursor="hand2",command= Negative).place(x=30, y=125)
Button( ws,text='Log Image',width=12,font=("ariel 17 bold"),bg='#3498DB',cursor="hand2",command= log).place(x=225, y=12)
Button( ws,text='power low',width=12,font=("ariel 17 bold"),bg='#3498DB',cursor="hand2",command= power).place(x=225, y=70)
Button( ws,text='Median Filter',width=12,font=("ariel 17 bold"),bg='#3498DB',cursor="hand2",command= medianFilter).place(x=225, y=125)
Button( ws,text='GS Approch1',width=12,font=("ariel 17 bold"),bg='#3498DB',cursor="hand2",command= GSApproch1).place(x=420, y=12)
Button( ws,text='GS Approch2',width=12,font=("ariel 17 bold"),bg='#3498DB',cursor="hand2",command= GSApproch2).place(x=420, y=70)
Button( ws,text='Avarage',width=12,font=("ariel 17 bold"),bg='#3498DB',cursor="hand2",command= Avarage).place(x=420, y=125)
Button( ws,text='Adding',width=12,font=("ariel 17 bold"),bg='#3498DB',cursor="hand2",command= add).place(x=615, y=12)
Button( ws,text='Gray Image',width=12,font=("ariel 17 bold"),bg='#3498DB',cursor="hand2",command= GrayImage).place(x=615, y=70)
Button( ws,text='Sharping',width=12,font=("ariel 17 bold"),bg='#3498DB',cursor="hand2",command= sharp).place(x=615, y=125)

canvas2 = Canvas(ws, width="400", height="400", relief=RIDGE, bd=2)
canvas2.place(x=10, y=230)
canvas3 = Canvas(ws, width="400", height="400", relief=RIDGE, bd=2)
canvas3.place(x=420, y=230)

var1 = StringVar()
label1 = Label( ws, textvariable=var1, relief=RAISED,font=("ariel 17 bold") ,bg="#17202A",fg="#F0F3F4")
var1.set("Origional Image")
label1.place(x=100, y=190)

var2 = StringVar()
label2 = Label( ws, textvariable=var2, relief=RAISED,font=("ariel 17 bold"),bg="#17202A",fg="#F0F3F4" )
var2.set("Second Image")
label2.place(x=540, y=190)

Directory = StringVar()
Label(ws,text='Image path:',font="ariel 15 bold",bg="#17202A",fg="#F0F3F4").place(x=20,y=650)
entry = Entry(ws, font="ariel 15",textvariable=Directory,width=30).place(x=140,y=650)

values = [1, 2, 3, 4, 5, 6, 7, 8]
scale_n = ttk.Combobox(ws, values=values, font=('ariel 15 bold'),width=2)
scale_n.insert('0', values[0])
scale_n.place(x=520, y=650)
Label(ws, text="K:", font=("ariel 15 bold"),bg="#17202A",fg="#F0F3F4").place(x=490, y=650)

values2 = [10, 30, 50, 80 , 100]
Label(ws,text='Add:',font="ariel 15 bold",bg="#17202A",fg="#F0F3F4").place(x=580,y=650)
entry2 = ttk.Combobox(ws, font="ariel 15",values=values2,width=3)
entry2.place(x=630,y=650)
entry2.insert('0', values2[0])

values3 = [0.1, 0.3, 0.5, 0.8, 1.2, 1.5, 2]
Label(ws,text='Gamma:',font="ariel 15 bold",bg="#17202A",fg="#F0F3F4").place(x=695,y=650)
entry3 = ttk.Combobox(ws, font="ariel 15",values=values3,width=3)
entry3.place(x=780,y=650)
entry3.insert('0', values3[0])

ws.mainloop()


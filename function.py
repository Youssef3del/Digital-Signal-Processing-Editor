import cv2 as cv
from cv2 import *
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageTk, ImageFilter, ImageEnhance, ImageOps
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter.filedialog import askopenfilename,asksaveasfilename
from math import sqrt
from math import exp
from math import * 

def GrayImage(image_path):
    img = cv.imread(image_path)
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return imgGray

def Negative(image_path):
    n = 8
    l = (pow(2,n)-1)
    img = GrayImage(image_path)
    # Rows = img.shape[0]
    # Columns = img.shape[1]
    # # Dimentions = img.shape[2]
    # negative = np.array(img,dtype='uint8')
    # # for d in range(Dimentions):
    # for r in range(Rows):
    #   for c in range(Columns):
    #       negative[r][c] = l-img[r][c]     
    negative = np.array(l-img, dtype = 'uint8')         
    return negative 

def GrayScaling(image_path,scale):
    img = GrayImage(image_path)
    k=scale
    target_lvl = 2**k
    target_compr_fargot = 256/target_lvl
    red_img = np.floor(img/256*target_lvl)
    GrayScaling = np.array(red_img*target_compr_fargot,dtype= 'uint8')

    # Rows = img.shape[0]
    # Columns = img.shape[1]
    # GrayScaling = np.array(img,dtype='uint8')
    # for r in range(Rows):
    #   for c in range(Columns):
    #       if img[r][c]>=128:
    #         GrayScaling[r][c] = 255
    #       else :
    #           GrayScaling[r][c] = 0
    return GrayScaling      

def BitPlaneSlicing(image_path):
    imgGray = cv.imread(image_path,0)
    fig = plt.figure(figsize=(15,5))
    fig.add_subplot(2,5,1)
    plt.imshow(imgGray,cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    Rows = imgGray.shape[0]
    Columns = imgGray.shape[1]
    bitImg =[]
    for i in range(Rows):
      for j in range(Columns):
        bitImg.append(np.binary_repr(imgGray[i][j],width=8))
    for k in range(8):
      k2 = 7
      k_bit_img = (np.array([int(i[k]) for i in bitImg],dtype='uint8')*(2**k2)).reshape(Rows,Columns)
      fig.add_subplot(2,5,k+2)
      plt.imshow(k_bit_img,cmap='gray')
      plt.title("{}-bit image".format(abs(k-8)))
      plt.axis('off')
      k2-=1   

def contrast(image_path):
    # img = cv.imread(image_path)
    img = GrayImage(image_path)
    s1 = 0
    s2 = 255
    r1 = img.min()
    r2 = img.max()
    contrast = np.array(((s2 - s1)/(r2 - r1)) * (img - r1) + s1, dtype = 'uint8')
    return contrast         

def GSApproch1(image_path,GS_from,GS_to):
    img = GrayImage(image_path)
    Rows = img.shape[0]
    Columns = img.shape[1]
    GSApproch1 = np.array(img,dtype='uint8')
    for r in range(Rows):
      for c in range(Columns):
        if img[r][c]>=GS_from and img[r][c]<=GS_to:
            GSApproch1[r][c] = 255
        else :
            GSApproch1[r][c] = 0       
    # GSApproch1 = np.zeros(img)
    # GSApproch1 = np.where(img in range(100,200,1),255,img)
    # GSApproch1 = np.array(GSApproch1,dtype='uint8')
    return GSApproch1       

def GSApproch2(image_path,GS_from,GS_to):
    imgGray = GrayImage(image_path)
    Rows = imgGray.shape[0]
    Columns = imgGray.shape[1]
    GSApproch2 = np.array(imgGray,dtype='uint8')
    for r in range(Rows):
      for c in range(Columns):
        if imgGray[r][c]>=GS_from and imgGray[r][c]<=GS_to:
           GSApproch2[r][c] = 255
        else :
           GSApproch2[r][c] = imgGray[r][c]          
    return GSApproch2

def add(image_path,x):
    img = GrayImage(image_path)
    out = img+x
    output = np.where(out>255,255,out)
    output = np.array(output,dtype='uint8')
    return output

def log(image_path):
    img = GrayImage(image_path)
    # Rows = img.shape[0]
    # Columns = img.shape[1]
    # # Dimentions = img.shape[2]
    # log = np.array(img,dtype='uint8')
    # # img = im2double(img)
    # # for d in range(Dimentions):
    # for r in range(Rows):
    #   for c in range(Columns):
    #       log[r][c] =100*math.log(1+img[r][c])
    # # log = round(3*np.log2(img))
    
    c = 255 / np.log(1 + np.max(img))
    log_image = c * (np.log(img + 1))
    log_image = np.array(log_image, dtype = np.uint8)
    return log_image    
    
def power(image_path,gamma):
    img = GrayImage(image_path)
    # Rows = img.shape[0]
    # Columns = img.shape[1]
    # # Dimentions = img.shape[2]
    # power = np.array(img,dtype='uint8')
    # # for d in range(Dimentions):
    # for r in range(Rows):
    #   for c in range(Columns):
    #       power[r][c] =255*(img[r][c] / 255) **.1     
    
    power = np.array(255*(img/ 255) **gamma, dtype = 'uint8')
    return power          

def inpower(gamma,image_path):
    img = GrayImage(image_path)
    gamma_corrected = np.array(255*(img / 255) **gamma, dtype = 'uint8')
    
    fig = plt.figure(figsize=(5,5))
    fig.add_subplot(1,2,1)
    plt.imshow(img,cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    fig.add_subplot(1, 2, 2)
    plt.imshow(gamma_corrected, cmap='gray')
    plt.title('invers power {}'.format(gamma))
    plt.axis('off')  
    
def Avarage(image_path):
    # img = cv.imread(image_path)
    img = GrayImage(image_path)
    rows = img.shape[0]
    columns = img.shape[1]
    output = np.array(img,dtype='uint8')
    for i in range(1,rows-1):
      for j in range(1,columns-1):
        temp =np.array([
               img[i-1,j-1],img[i-1,j],img[i-1,j+1],
               img[i,j-1],img[i,j],img[i,j+1],
               img[i+1,j-1],img[i+1,j],img[i+1,j+1]
               ])
        output[i][j]=np.round(np.sum(temp)/9)
    return output       
    
def medianFilter(image_path):
    img = GrayImage(image_path)
    rows = img.shape[0]
    columns = img.shape[1]
    output = np.array(img,dtype='uint8')
    for i in range(1,rows-1):
      for j in range(1,columns-1):
        temp =[
              img[i-1,j-1],img[i-1,j],img[i-1,j+1],
               img[i,j-1],img[i,j],img[i,j+1],
               img[i+1,j-1],img[i+1,j],img[i+1,j+1]
               ]
        temp = np.sort(temp)
        output[i][j]=temp[4]          
    return output   

def max_filter(image_path):
    img = GrayImage(image_path)
    rows = img.shape[0]
    columns = img.shape[1]
    output = np.array(img,dtype='uint8')
    for i in range(1,rows-1):
      for j in range(1,columns-1):
        temp =[
              img[i-1,j-1],img[i-1,j],img[i-1,j+1],
               img[i,j-1],img[i,j],img[i,j+1],
               img[i+1,j-1],img[i+1,j],img[i+1,j+1]
               ]
        temp = np.sort(temp)
        output[i][j]=temp.max()          
    fig = plt.figure(figsize=(5,5))
    fig.add_subplot(1,2,1)
    plt.imshow(img,cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    fig.add_subplot(1, 2, 2)
    plt.imshow(output, cmap='gray')
    plt.title('Max filter')
    plt.axis('off')   
    
def min_filter(image_path):
    img = GrayImage(image_path)
    rows = img.shape[0]
    columns = img.shape[1]
    output = np.array(img,dtype='uint8')
    for i in range(1,rows-1):
      for j in range(1,columns-1):
        temp =[
              img[i-1,j-1],img[i-1,j],img[i-1,j+1],
               img[i,j-1],img[i,j],img[i,j+1],
               img[i+1,j-1],img[i+1,j],img[i+1,j+1]
               ]
        temp = np.sort(temp)
        output[i][j]=temp.min()          
    fig = plt.figure(figsize=(5,5))
    fig.add_subplot(1,2,1)
    plt.imshow(img,cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    fig.add_subplot(1, 2, 2)
    plt.imshow(output, cmap='gray')
    plt.title('Min filter')
    plt.axis('off')       


def sharp(image_path):   
    img = GrayImage(image_path)
    mask = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    
    # st = time.time()
    # rows = img.shape[0]
    # columns = img.shape[1]
    # out = np.array(img,dtype='uint8')
    # for i in range(1,rows-1):
    #     for j in range(1,columns-1):
    #         temp = mask*img[i-1:i+2,j-1:j+2]
    #         value = np.sum(temp)
    #         out[i,j] = abs(value)  
    # sharp = np.array(out,dtype='uint8')
    # time1 = (time.time()- st)
    # print("[*] Sharping1 process time : %f" % time1)
    
    # # shapes
    # st2 = time.time()
    # Hi, Wi = img.shape
    # Hk, Wk = mask.shape
    # hk = Hk//2
    # wk = Wk//2
    # # padding
    # new_img = np.pad(img, (hk, wk), 'constant', constant_values=0)
    # pHi, pWi = new_img.shape
    # sharp = np.zeros((Hi, Wi))
    # for i in range(hk, pHi-hk):
    #     for j in range(wk, pWi-wk):
    #         batch = new_img[i-hk:i+hk+1, j-wk:j+wk+1]
    #         sharp[i-hk][j-wk] = np.sum(batch*mask)
    # time2 = (time.time()- st2)        
    # print("[*] Sharping2 process time : %f" % time2)
    
    st3 = time.time()
    sharp = np.array(img,dtype='uint8')
    sharp = cv.filter2D(img ,dst = img.shape,kernel = mask,ddepth=-1)
    time3 = (time.time()- st3)
    print("[*] Sharping3 process time : %f" % time3)
    # print("[*] performance is : {}".format(time3/time1))
    return sharp

def first_drev(image_path):
    img = GrayImage(image_path)
    rows = img.shape[0]
    columns = img.shape[1]
    output = np.array(img,dtype='uint8')
    for i in range(1,rows-1):
      for j in range(1,columns-1):
        temp =[
              img[i-1,j-1],img[i-1,j],img[i-1,j+1],
               img[i,j-1],img[i,j],img[i,j+1],
               img[i+1,j-1],img[i+1,j],img[i+1,j+1]
               ]
        Gx = abs(temp[0]*-1+ temp[1]*-1+ temp[2]*-1+temp[6]+temp[7]+temp[8])
        Gy = abs(temp[0]*-1+ temp[3]*-1+ temp[6]*-1+temp[6]+temp[7]+temp[8])
        output[i][j] = Gx+Gy 
        
    fig = plt.figure(figsize=(5,5))
    fig.add_subplot(1,2,1)
    plt.imshow(img,cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    fig.add_subplot(1, 2, 2)
    plt.imshow(output, cmap='gray')
    plt.title('first drev filter')
    plt.axis('off')    

def histogram(image_path):
    img = cv.imread(image_path,0)
    # img = GrayImage(image_path)
    hist = plt.hist(img.ravel(),256,[0,256])
    plt.hist(hist)
    plt.show()
    
def HistogramEqualization(image_path):
    Gimg = cv.imread(image_path, 0)
    img = cv.imread(image_path)
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(1, 2, 1)
    plt.imshow(img[:,:,[2,1,0]])
    plt.title('Original Image')
    plt.axis('off')
    
    rows = Gimg.shape[0]
    columns = Gimg.shape[1]
    number_of_pixels = rows * columns
    
    Histogram_Image = np.zeros((rows, columns),dtype='uint8')
    
    freq = np.zeros((256,1))
    prob = np.zeros((256,1))
    output = np.zeros((256,1))
    
    for i in range(rows):
        for j in range(columns):
            freq[Gimg[i][j]] += 1
            
    sum = 0
    no_bins = 255
    
    for i in range(freq.size):
        sum+=freq[i]
        prob[i] = sum/number_of_pixels
        output[i]=np.round(prob[i]*no_bins)
        
    for i in range(rows):
        for j in range(columns):
            Histogram_Image[i][j] = output[Gimg[i][j]]
            
    fig.add_subplot(1, 2, 2)
    plt.imshow(Histogram_Image, cmap='gray')
    plt.title('Histogram Equalization')
    plt.axis('off')     
      
def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def gaussianHP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base    

def gaussianLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

def try_d0s_lp(d0,image_path):
    
    img = GrayImage(image_path)
    plt.figure(figsize=(25, 5), constrained_layout=False)


    plt.subplot(161), plt.imshow(img, "gray"), plt.title("Original Image")

    original = np.fft.fft2(img)
    plt.subplot(162), plt.imshow(np.log(1+np.abs(original)), "gray"), plt.title("Spectrum")

    center = np.fft.fftshift(original)
    plt.subplot(163), plt.imshow(np.log(1+np.abs(center)), "gray"), plt.title("Centered Spectrum")

    LowPassCenter = center * gaussianLP(d0,img.shape)
    plt.subplot(164), plt.imshow(np.log(1+np.abs(LowPassCenter)), "gray"), plt.title("Centered Spectrum multiply Low Pass Filter")

    LowPass = np.fft.ifftshift(LowPassCenter)
    plt.subplot(165), plt.imshow(np.log(1+np.abs(LowPass)), "gray"), plt.title("Decentralize")

    inverse_LowPass = np.fft.ifft2(LowPass)
    plt.subplot(166), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Processed Image")

    plt.suptitle("D0:"+str(d0),fontweight="bold")
    plt.subplots_adjust(top=1.1)
    plt.show()

def try_d0s_hp(d0,image_path):
    
    img = GrayImage(image_path)
    plt.figure(figsize=(25, 5), constrained_layout=False)


    plt.subplot(161), plt.imshow(img, "gray"), plt.title("Original Image")

    original = np.fft.fft2(img)
    plt.subplot(162), plt.imshow(np.log(1+np.abs(original)), "gray"), plt.title("Spectrum")

    center = np.fft.fftshift(original)
    plt.subplot(163), plt.imshow(np.log(1+np.abs(center)), "gray"), plt.title("Centered Spectrum")
 
    HighPassCenter = center * gaussianHP(d0,img.shape)
    plt.subplot(164), plt.imshow(np.log(1+np.abs(HighPassCenter)), "gray"), plt.title("Centered Spectrum multiply High Pass Filter")

    HighPass = np.fft.ifftshift(HighPassCenter)
    plt.subplot(165), plt.imshow(np.log(1+np.abs(HighPass)), "gray"), plt.title("Decentralize")

    inverse_HighPass = np.fft.ifft2(HighPass)
    plt.subplot(166), plt.imshow(np.abs(inverse_HighPass), "gray"), plt.title("Processed Image")

    plt.suptitle("D0:"+str(d0),fontweight="bold")
    plt.subplots_adjust(top=1.1)
    plt.show()
    
def And(image_path):
    img = GrayImage(image_path)
    rows = img.shape[0]
    columns = img.shape[1]
    segment = np.zeros((rows,columns),dtype='uint8')
    for i in range(500,1000):
      for j in range(500,1000):
          segment[i][j]=1
    output = np.array(img,dtype='uint8')
    for i in range(1,rows-1):
      for j in range(1,columns-1): 
          if(segment[i][j] and img[i][j]):
              output[i][j]= img[i][j]
          else :
              output[i][j]= 0
              
    fig = plt.figure(figsize=(5,5))
    fig.add_subplot(1,2,1)
    plt.imshow(img,cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    fig.add_subplot(1, 2, 2)
    plt.imshow(output, cmap='gray')
    plt.title('And')
    plt.axis('off')           
    
def Or(image_path):
    img = GrayImage(image_path)
    rows = img.shape[0]
    columns = img.shape[1]
    segment = np.ones((rows,columns),dtype='uint8')
    for i in range(500,1000):
      for j in range(500,1000):
          segment[i][j]=0
    output = np.array(img,dtype='uint8')
    for i in range(1,rows-1):
      for j in range(1,columns-1): 
          if(segment[i][j] and img[i][j]):
              output[i][j]= 255
          else :
              output[i][j]= img[i][j]
              
    fig = plt.figure(figsize=(5,5))
    fig.add_subplot(1,2,1)
    plt.imshow(img,cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    fig.add_subplot(1, 2, 2)
    plt.imshow(output, cmap='gray')
    plt.title('Or')
    plt.axis('off')    
    
# Or("C:\\Users\\youss\\DIP_Project\\images\\arya_stark.jpg")   
# And("C:\\Users\\youss\\DIP_Project\\images\\arya_stark.jpg")    
# first_drev("C:\\Users\\youss\\DIP_Project\\images\\arya_stark.jpg")    
# min_filter("C:\\Users\\youss\\DIP_Project\\images\\arya_stark.jpg")
# max_filter("C:\\Users\\youss\\DIP_Project\\images\\arya_stark.jpg")
# inpower(2,"C:\\Users\\youss\\DIP_Project\\images\\arya_stark.jpg")
# BitPlaneSlicing("C:\\Users\\youss\\DIP_Project\\images\\arya_stark.jpg")
# try_d0s_hp(30,"C:\\Users\\youss\\DIP_Project\\images\\arya_stark.jpg")    
# try_d0s_lp(10,"C:\\Users\\youss\\DIP_Project\\images\\arya_stark.jpg") 
# histogram("C:\\Users\\youss\\DIP_Project\\images\\arya_stark.jpg") 
# HistogramEqualization("C:\\Users\\youss\\DIP_Project\\images\\arya_stark.jpg")
import numpy as np
#import matplotlib.pyplot as plt
import cv2  # OpenCV library for computer vision
#from PIL import Image
import PyPDF2
from pdf2image import convert_from_bytes
import io
import os
import glob
from sklearn.svm import SVC
#from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
#from sklearn.model_selection import train_test_split
from sklearn.decomposition import RandomizedPCA

def pdf2img(pdf_object):
    
    dst_pdf = PyPDF2.PdfFileWriter()
    dst_pdf.addPage(pdf_object.getPage(0))

    pdf_bytes = io.BytesIO()

    dst_pdf.write(pdf_bytes)

    return convert_from_bytes(pdf_bytes.getvalue())

def PDFsplit(pdf):
    # creating input pdf file object
    pdfFileObj = open(pdf, 'rb')
     
    # creating pdf reader object
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

    folder=pdf.split('.pdf')[0].split('/')[1]
 
    for page in range(pdfReader.numPages):

        # creating pdf writer object for (i+1)th split
        pdfWriter = PyPDF2.PdfFileWriter()
         
        # output pdf file name
        #outputpdf =  'Pheonix/'+folder+'/'+folder + '_'+str(i) + '.pdf'

        # adding pages to pdf writer object
        pdfWriter.addPage(pdfReader.getPage(page))
        img=pdf2img(pdfWriter) 
        #print(np.array(img[0]))
        img= cv2.cvtColor(np.array(img[0]), cv2.COLOR_BGR2RGB)
        img= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        iimg=cv2.GaussianBlur(img,(5,5),1)
        img=cv2.GaussianBlur(img,(5,5),1)
        img=cv2.GaussianBlur(img,(5,5),1)
        img=cv2.GaussianBlur(img,(5,5),1)
        img=cv2.resize(img,(147,103))
        img=cv2.GaussianBlur(img,(5,5),1)
        img=cv2.GaussianBlur(img,(5,5),1)
        img=cv2.GaussianBlur(img,(5,5),1)
        img=cv2.GaussianBlur(img,(5,5),1)
        img=img.flatten()
        img=img.reshape(1, -1)
        
        clf= joblib.load('model/svc.pkl')
        pca=joblib.load('model/pca.pkl')
        img=pca.transform(img)
        X_train_pca_a=[]
        A=27
        B=A+2
        X_test_pca_a=[]
        for li in img:
            temp1=li[A:B].copy()
            temp2=li[:18].copy()
            X_train_pca_a.append(np.concatenate((temp1, temp2)))
        pred=clf.predict(X_train_pca_a)
        prob=max(clf.predict_proba(X_train_pca_a)[0])
        if prob <.4: 
            #print(pred[0],prob)
            file='other'
        else:
            #print(pred[0])
            file=pred[0]
        #print(os.getcwd())
        #check if that folder is created if not then create and save pdf to that folder
        folder=os.getcwd()+'/'+file
        #print(folder)
        if not os.path.exists(folder):
            os.makedirs(file)
            
        with open(file+'/'+str(page)+'.pdf', "wb") as f:
                pdfWriter.write(f)  
 

    pdfFileObj.close()

#give the path to pdf file
pdffile_path='/Users/task/ai_projects/orodoc/Pheonix2/00010808.pdf'
PDFsplit(pdffile_path)

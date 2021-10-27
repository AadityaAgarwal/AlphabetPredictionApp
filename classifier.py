import cv2
import numpy as np
import pandas as pd
from pandas.core import frame
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_openml as fo
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score 
from PIL import Image
import PIL.ImageOps
import os,time,ssl


X=np.load('image.npz')['arr_0']
y=pd.read_csv('labels.csv')['labels']
print(pd.Series(y).value_counts())
classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses=len(classes)

x_test,x_train,y_test,y_train=tts(X,y,random_state=9,train_size=7500,test_size=2500)
x_trainScale=x_train/255.0
x_testScale=x_test/255.0
clf=lr(solver='saga',multi_class="multinomial").fit(x_trainScale,y_train)
y_pred=clf.predict(x_testScale)
print(f"Accuracy:{accuracy_score(y_test,y_pred)*100}")

cap=cv2.VideoCapture(0)

while True:
    try:
        ret,frame=cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width=gray.shape
        upper_left=(int(width/2-56),int(height/2-56))
        bottom_right=(int(width/2+56),int(height/2+56))
        cv2.rectangle(gray,upper_left,bottom_right,(0,255,0),2)
        roi=gray[upper_left[1]:bottom_right[1],upper_left[0],bottom_right[0]]
        img_pil=Image.fromarray(roi)
        image_pw=img_pil.convert('L')
        image_pw_resized=image_pw.resize((28,28),Image.ANTIALIAS)
        image_pw_resized_inverted=PIL.ImageOps.invert(image_pw_resized)
        pixel_filter=20
        minPixel=np.percentile(image_pw_resized_inverted,pixel_filter)
        image_pw_resized_inverted_scaled=np.clip(image_pw_resized_inverted-minPixel,0,255)
        maxPixel=np.max(image_pw_resized_inverted)
        image_pw_resized_inverted_scaled=np.asarray(image_pw_resized_inverted_scaled)/maxPixel
        testSample=np.array(image_pw_resized_inverted_scaled).reshape(1,784)
        testPredict=clf.predict(testSample)
        print(testPredict)
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    except Exception as e:
        pass
cap.release()
cv2.destroyAllWindows()
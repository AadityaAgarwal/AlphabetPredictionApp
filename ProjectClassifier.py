# from numpy.random.mtrand import multinomial
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.datasets import fetch_openml as fo
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as lr
from PIL import Image
import PIL.ImageOps

X=np.load('image.npz')['arr_0']
y=pd.read_csv('labels.csv')['labels']
print(pd.Series(y).value_counts())
classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclass=len(classes)

x_test,x_train,y_test,y_train=tts(X,y,random_state=9,train_size=3500,test_size=500)
x_trainScale=x_train/255.0
x_testScale=x_test/255.0

clf=lr(solver='saga',multi_class="multinomial").fit(x_trainScale,y_train)


def getPrediction(image):
    img_pil=Image.open(image)
    img_pxl=img_pil.convert('L')
    img_pxl_resized=img_pxl.resize((28,28),Image.ANTIALIAS)
    pixel_filter=20
    min_pxl=np.percentile(img_pxl_resized,pixel_filter)
    img_pxl_resized_inverted_scaled=np.clip(img_pxl_resized-min_pxl,0,255)
    max_pxl=np.max(img_pxl_resized)
    img_pxl_resized_inverted_scaled=np.asarray(img_pxl_resized_inverted_scaled/max_pxl)

    testSample=np.array(img_pxl_resized_inverted_scaled).reshape(1,784)
    testPredict=clf.predict(testSample)

    return testPredict[0]
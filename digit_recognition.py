import cv2
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps 
import os,ssl,time

# Setting an https context to fetch the data from openml

if(not os.environ.get('PYTHONHTTPSVERIFY','') and getattr(ssl,'_create_unverified_context', None)):
    ssl._create_default_https_context=ssl._create_unverified_context



# Fetching the data

X,y=fetch_openml('mnist_784',version=1,return_X_y=True)
print(pd.Series(y).value_counts())
classes=['0','1','2','3','4','5','6','7','8','9']
n_classes=len(classes)

# Splitting the data and scaling it

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=9,train_size=7500,test_size=2500)
X_train_scale=X_train/255.0
X_test_scale=X_test/255.0

lr=LogisticRegression(solver='saga',multi_class='multinomial').fit(X_train_scale,y_train)
y_pred=lr.predict(X_test_scale)
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)

# Starting the camera

cap=cv2.VideoCapture(0)
while True:
    try:
        ret,frame=cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # Drawing a box in the center of the video

        height,width=gray.shape
        upper_left=(int(width/2-56),int(height/2-56))
        bottom_right=(int(width/2+56),int(height/2+56))
        cv2.rectangle(gray,upper_left,bottom_right,(0,255,0),2)

        # Detecting the digit inside the box

        roi=gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]

        # Converting the image into PIL format

        im_pil=Image.fromarray(roi)

        # Converting to greyscale image

        image_bw=im_pil.convert('L')
        image_bw_resize=image_bw.resize((28,28),Image.ANTIALIAS)

        # Invert the image

        image_bw_resize_inverted=PIL.ImageOps.invert(image_bw_resize)
        pixel_filter=20

        # Converting the scaler quantity

        minimum_pixel=np.percentile(image_bw_resize_inverted,pixel_filter)

        # Limit the values between 0 and 255

        image_bw_resize_inverted_scale=np.clip(image_bw_resize_inverted-minimum_pixel,0,255)
        max_pixel=np.max(image_bw_resize_inverted)

        # Converting to array

        image_bw_resize_inverted_scale=np.asarray(image_bw_resize_inverted_scale/max_pixel)

        # Creating a test sample and making a prediction

        test_sample=np.array(image_bw_resize_inverted_scale).reshape(1784)
        test_pred=lr.predict(test_sample)
        print('Predicted class is:', test_pred)

        # Display the resulting frame

        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    except Exception as e:
        pass
    cap.release()
    cv2.destroyAllWindows()

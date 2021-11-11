import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import cv2

from PIL import Image 
import PIL.ImageOps
import os, time

x,y=fetch_openml('mnist_784',version=1,return_X_y=True)
print(pd.Series(y).value_counts())

classes=['0','1','2','3','4','5','6','7','8','9']
numberOfClasses=10

XTrain,XTest,YTrain,YTest=train_test_split(x,y,train_size=7500,test_size=2500,random_state=9)
XTrainScaled=XTrain/255
XTestScaled=XTest/255

classifier=LogisticRegression(solver='saga',multi_class='multinomial')
classifier.fit(XTrainScaled,YTrain)

ypredict=classifier.predict(XTestScaled)
accuracy=accuracy_score(YTest,ypredict)
print(accuracy)

cap =cv2.VideoCapture(0)

while(True):
    try:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        upper_left = (int(width /2 - 56), int(height/2 - 56))
        lower_right = (int(width /2 + 56), int(height/2 + 56))
        cv2.rectangle(gray, upper_left, lower_right, (0, 255, 0), 2)
        #roi = region of interest

        roi = gray[upper_left[1]: lower_right[1], upper_left[0]:lower_right[0]]

        image_pil = Image.fromarray(roi)

        image_pil_gray = image_pil.convert('L')

        image_pil_gray_resize = image_pil_gray.resize((28,28), Image.ANTIALIAS)

        image_pil_gray_resize_inverted = PIL.ImageOps.invert(image_pil_gray_resize)

        pixel_filter = 20

        min_pixel = np.percentile(image_pil_gray_resize_inverted, pixel_filter)

        image_pil_gray_resize_inverted_scaled = np.clip(image_pil_gray_resize_inverted - min_pixel, 0, 255)

        max_pixel = np.max(image_pil_gray_resize_inverted)

        image_pil_gray_resize_inverted_scaled = np.asarray(image_pil_gray_resize_inverted_scaled) / max_pixel

        test_sample = np.array(image_pil_gray_resize_inverted_scaled).reshape(1,784)

        test_prediction = classifier.predict(test_sample)
                  
        print("The predicted class is ", test_prediction)

        cv2.imshow('frame', gray)

    except Exception as e:
        pass

cap.release()

cv2.destroyAllWindows()

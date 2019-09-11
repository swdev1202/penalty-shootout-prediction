from keras.models import load_model
from keras.preprocessing.image import img_to_array
import os
import numpy as np
import cv2
import time

model_dir = 'model'
model_name = 'model_complete.h5'

# load model
model = load_model(os.path.join(model_dir, model_name))

image_path = 'data/test'
image_names = os.listdir(image_path)

for image in image_names:
    start = time.time()
    img = cv2.imread(os.path.join(image_path,image), 3)
    #cv2.imshow("img", img)
    #cv2.waitKey(0)

    img = img.astype('float')/255.0
    
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    pred = pred.argmax(axis=1)[0]
    
    label = ""
    if(pred == 0):
        label = "center"
    elif(pred == 1):
        label = "left"
    elif(pred == 2):
        label = "right"
    else:
        label = "unknown"

    end = time.time()

    img = cv2.imread(os.path.join(image_path,image), 3)
    cv2.putText(img, label, (3,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    time_diff = str(end-start) + " sec"
    cv2.putText(img, time_diff, (180,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    cv2.imshow("img", img)
    cv2.waitKey(0)
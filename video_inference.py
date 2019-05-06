from keras.models import load_model
from keras.preprocessing.image import img_to_array
import os
import numpy as np
import cv2
import time

Y_MAX = 564
Y_MIN = 308
X_MAX = 706
X_MIN = 450

model_dir = 'model'
model_name = 'model_256_256_20epoch_entire_model.h5'

# load model
model = load_model(os.path.join(model_dir, model_name))

video_path = 'data/video'
video_name = 'test.mp4'

vid = cv2.VideoCapture(os.path.join(video_path, video_name))
vid.set(cv2.CAP_PROP_POS_MSEC,33.3)

index = 0
while(True):
    ret, frame = vid.read()
    
    if not ret:
        break
    
    # frame.shape = (720, 1280, 3)
    # lets cropped the one we want
    crop = frame[Y_MIN:Y_MAX, X_MIN:X_MAX]

    img = crop.astype('float')/255.0
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

    cv2.putText(frame, label, (500,260), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0,0,255), 2)
    
    #name = './data/video_results' + str(index) + '.jpg'
    #cv2.imwrite(name, frame)
    cv2.imshow("output", frame)
    cv2.waitKey(0)
    index += 1
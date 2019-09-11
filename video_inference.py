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

LEFT_MIN = (280,141)
LEFT_MAX = (522,351)

CENTER_MIN = (522,141)
CENTER_MAX = (764,351)

RIGHT_MIN = (764,141)
RIGHT_MAX = (1006,351)

model_dir = 'model'
model_name = 'model_complete.h5'

# load model
model = load_model(os.path.join(model_dir, model_name))

video_path = 'data/video'
video_name = 'test.mp4'

vid = cv2.VideoCapture(os.path.join(video_path, video_name))
vid.set(cv2.CAP_PROP_POS_MSEC,33.3)

#fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#out = cv2.VideoWriter('output.avi', fourcc, 60.0, (1280,720))

index = 0
while(True):
    start = time.time()
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
        #cv2.rectangle(frame, CENTER_MIN, CENTER_MAX, (255,0,0), 1)
    elif(pred == 1):
        label = "left"
        #cv2.rectangle(frame, LEFT_MIN, LEFT_MAX, (0,255,0), 1)
    elif(pred == 2):
        label = "right"
        #cv2.rectangle(frame, RIGHT_MIN, RIGHT_MAX, (0,0,225), 1)
    else:
        label = "unknown"
    end = time.time()

    time_diff = str(int(1/(end-start))) + " FPS"
    cv2.putText(frame, time_diff, (3,30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255), 2)
    cv2.putText(frame, label, (1150,30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255), 2)
    
    #out.write(frame)

    #name = '/result/video_frames' + str(index) + '.jpg'
    #cv2.imwrite(name, frame)
    cv2.imshow("output", frame)
    cv2.waitKey(0)
    #index += 1

#vid.release()
#out.release()
#cv2.destroyAllWindows()
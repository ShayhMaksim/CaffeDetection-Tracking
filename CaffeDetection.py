#Import the neccesary libraries
import numpy as np
import argparse
import cv2 
from models import *
import pandas as pd
import time

df=pd.DataFrame(columns=['C_X','C_Y','width','height','D','Vx','Vy','fps','try'])
print(df.head())
table_index=0
_try=0

# Labels of Network.
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

# Open video file or capture device. 
cap = cv2.VideoCapture("video2.avi")
#cap = cv2.VideoCapture(1)
#Load the Caffe model 
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

main_object=Class(None,None,None)
tracker = cv2.TrackerCSRT_create()
key=True
count_img=0
count_Mtemplate=0

# Default resolutions of the frame are obtained.The default resolutions are system dependent.

# We convert the resolutions from float to integer.

frame_width = int(cap.get(3))

frame_height = int(cap.get(4))

#ok=None

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.

out = cv2.VideoWriter('TrackerCSRT.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while True:
    # Start timer
    timer = cv2.getTickCount()
    # Capture frame-by-frame
    ok, frame = cap.read()
    mini_time=time.time()
    
    if (key==True):
        tracker = cv2.TrackerCSRT_create()
        
        frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction
        main_object=Class(None,None,None)

        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
        #Set to network the input blob 
        net.setInput(blob)
        #Prediction of network
        detections = net.forward()

        #Size of frame resize (300x300)
        cols = frame_resized.shape[1] 
        rows = frame_resized.shape[0]

    #For get the class and location of object detected, 
    # There is a fix index for class, location and confidence
    # value in @detections array .
        count_object=0

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2]) #Confidence of prediction 
            class_id = int(detections[0, 0, i, 1]) # Class label
            if (confidence > 0.4 and class_id==7): # Filter prediction 

            # Object location 
                xLeftBottom = int(detections[0, 0, i, 3] * cols) 
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop   = int(detections[0, 0, i, 5] * cols)
                yRightTop   = int(detections[0, 0, i, 6] * rows)
            
            # Factor for scale to original size of frame
                heightFactor = frame.shape[0]/300.0  
                widthFactor = frame.shape[1]/300.0 
            # Scale object detection to frame
                xLeftBottom = int(widthFactor * xLeftBottom) 
                yLeftBottom = int(heightFactor * yLeftBottom)
                xRightTop   = int(widthFactor * xRightTop)
                yRightTop   = int(heightFactor * yRightTop)
            # Draw location of object  
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),(0, 255, 0))

                #info=getAngle(Camera(640,480,320,240,pi*75./180),Object(4,1.5),camera_matrix,xLeftBottom,yLeftBottom,xRightTop,yRightTop)

            # Draw label and confidence of prediction in frame resized
                if class_id in classNames:
                    count_object=count_object+1

                    if xLeftBottom<0:
                        xLeftBottom=0
                    if yLeftBottom<0:
                        yLeftBottom=0
                    bbox = ((xLeftBottom), (yLeftBottom),abs(xRightTop-xLeftBottom), abs(yRightTop-yLeftBottom))
                    
                    # ok = tracker.init(frame, bbox)
                    
                    # ok, bb = tracker.update(frame)
                    if main_object.probability<confidence:
                        #main_object.info=info                      
                        main_object.map=bbox
                        main_object.name=classNames[class_id]
                        main_object.probability=confidence
                        ok = tracker.init(frame, bbox)

                    label = classNames[class_id] + ": " + str(confidence)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                     (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                     (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                    if ok==True:
                        key=False

        
  
    if key==False:
        #if count_Mtemplate!=0:
        ok, bbox = tracker.update(frame)

        # Draw bounding box
        if ok:
            count_Mtemplate=count_Mtemplate+1
            if (count_Mtemplate>120):
                current_img=frame[int(bbox[0]):int(bbox[0]+bbox[2]),int(bbox[1]):int(bbox[1]+bbox[3])]
                count_img=0
                _try=_try+1
                key=True
                cv2.putText(frame, "Update", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
                count_Mtemplate=0

            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

            xLeftBottom=int(bbox[0])
            yLeftBottom=int(bbox[1])
            xRightTop=int(bbox[2]+bbox[0])
            yRightTop=int(bbox[3]+bbox[1])
            info=getAngle(Camera(640,480,320,240,pi*75./180),Object(4,1.5),camera_matrix,xLeftBottom,yLeftBottom,xRightTop,yRightTop)


            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            cv2.putText(frame, "{0} ".format(main_object.name)+"{:.3f}".format(main_object.probability), (xLeftBottom, yLeftBottom),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            old_c_X,old_c_Y=main_object.getCenter()
            #центр цели по оси Х
            Center_x=bbox[0]+bbox[2]/2
            #центр цели по оси Y
            Center_y=bbox[1]+bbox[3]/2

            #print(time.time()-mini_time)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer); 
            Vx=float(Center_x-old_c_X)/(1./30)
            Vy=float(Center_y-old_c_Y)/(1./30)
            main_object.map=bbox
            main_object.info=info
            coeffVx,coeffVy=GetCoeff(main_object.info.D)
            main_object.Vx=Vx/fx*coeffVx
            main_object.Vy=Vy/fy*coeffVy

            df.loc[table_index]={'C_X':Center_x,'C_Y':Center_y,'width':bbox[2],'height':bbox[3],'D':main_object.info.D,'Vx':main_object.Vx,'Vy':main_object.Vy,'fps':(cv2.getTickFrequency() / (cv2.getTickCount() - timer)),'try':_try}
            table_index=table_index+1
        
            cv2.putText(frame, "D:"+"{:.3f}".format(main_object.info.D)+" "+\
                            "u_0:"+"{:.3f}".format(main_object.info.u_0)+" "+\
                            "fi_V:"+"{:.3f}".format(main_object.info.fi_v)+" "+\
                            "Vx:"+"{:.8f}".format(main_object.Vx)+" "+\
                            "Vy:"+"{:.8f}".format(main_object.Vy)+" ",\
                            (20, 40),\
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        else :
            # Tracking failure
            #count_img=count_img+1
            #if (count_img>3):
            #count_img=0
            _try=_try+1
            key=True
            cv2.putText(frame, "Tracking failure detected", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
            count_Mtemplate=0


    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer); 
    cv2.putText(frame,"fps:{0}".format(fps),(20, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    cv2.imshow("frame", frame)
    out.write(frame)
    
    if cv2.waitKey(10) >= 0:  # Break with ESC 
        break
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        # If the number of captured frames is equal to the total number of frames,
        # we stop
        break

df.head()
df.to_csv("TrackerCSRT")
cap.release()
out.release()
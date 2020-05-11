import numpy as np
import cv2

fx=7.8616850092106131e+02
fy=7.9401578293823570e+02

cx=320
cy=240

k1=1.1284230230538748e-01
k2=8.1798014396082175e-01
p=-8.7427103937023141e+00

camera_matrix = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]], dtype=np.float32)
distortion = np.array([k1, k2, 0, 0, p], dtype=np.float32)

mapx,mapy=cv2.initUndistortRectifyMap(camera_matrix, distortion, None, None, (640, 480), cv2.CV_16SC2)

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('video2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while True:
    ret, frame=cap.read()
    frame=cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT) 
    out.write(frame)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) >= 0:  # Break with ESC 
        cap.release()
        out.release()
        break

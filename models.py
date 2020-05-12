import numpy as np
import cv2
from math import *
fx=7.8616850092106131e+02
fy=7.9401578293823570e+02

cx=320
cy=240

k1=1.1284230230538748e-01
k2=8.1798014396082175e-01
p=-8.7427103937023141e+00

camera_matrix = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]], dtype=np.float32)
distortion = np.array([k1, k2, 0, 0, p], dtype=np.float32)


class BalisticInfo:
    def __init__(self,u_0,fi_v,D):
        self.u_0=u_0
        self.fi_v=fi_v
        self.D=D

class Camera:
     def __init__(self,width,height,c_X,c_Y,gamma):
        self.width=width
        self.height=height
        self.c_X=c_X
        self.c_Y=c_Y
        self.gamma=gamma


class Class:
    def __init__(self,info:BalisticInfo,name:str,map:tuple,Vx=0,Vy=0,probability=0.):
        self.info=info
        self.name=name
        self.map=map
        self.Vx=Vx
        self.Vy=Vy
        self.probability=probability

    def getCenter(self):
        #центр цели по оси Х
        Center_x=self.map[0]+self.map[2]/2
        #центр цели по оси Y
        Center_y=self.map[1]+self.map[3]/2
        return Center_x,Center_y

class Object:
    def __init__(self,width,height):
        self.width=width
        self.height=height

def getAngle(camera,RealObject,intrinsic_matrix,xLeftBottom,yLeftBottom,xRightTop,yRightTop):
    focus_x=intrinsic_matrix[0,0]
    focus_y=intrinsic_matrix[1,1]

    #угол обзора по ширине (половина угла)
    max_angle_x=atan(camera.width/(2*focus_x))
    #угол обзора по высоте
    max_angle_y=atan(camera.height/(2*focus_y))

    #ширина объекта в пикселя
    width_obj=abs(xRightTop-xLeftBottom)
    #высота объекта в пикселях
    height_obj=abs(yLeftBottom-yRightTop)

    #центр цели по оси Х
    Center_x=xLeftBottom+abs(width_obj)/2
    #центр цели по оси Y
    Center_y=yLeftBottom+abs(height_obj)/2

    #auto D_x=focus_x*RealObject.width/width_obj;
    if (height_obj<width_obj):
        D=focus_y*RealObject.height/(height_obj*sin(camera.gamma))
    else:
        D_x=focus_x*RealObject.width/(width_obj*sin(camera.gamma))
        D_y=focus_y*RealObject.height/(height_obj*sin(camera.gamma))
        D=(D_x+D_y)/2.

    u_0=max_angle_x*(Center_x-camera.c_X)/camera.c_X*180./pi
    fi_v=max_angle_x*(Center_y-camera.c_Y)/camera.c_Y*180./pi

    info=BalisticInfo(u_0,fi_v,D)

    return info
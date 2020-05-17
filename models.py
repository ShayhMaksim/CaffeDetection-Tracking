import numpy as np
import cv2
from math import *
fx=6.9642048339031021e+02
fy=6.9537324762648461e+02

cx=320
cy=240

k1=7.4349301802134074e-02
k2=-7.0749658329709775e-01
p=2.3059936688576812e+00
error=3.0413357901566346e-01
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
    def __init__(self,info:BalisticInfo,name:str,map:tuple,Vx=0.,Vy=0.,probability=0.):
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
        D=focus_y*RealObject.height/(height_obj)
    else:
        D_x=focus_x*RealObject.width/(width_obj)
        D_y=focus_y*RealObject.height/(height_obj)
        D=(D_x+D_y)/2.


    u_0=max_angle_y*(Center_y-camera.c_Y)/camera.c_Y*180./pi
    fi_v=max_angle_x*(Center_x-camera.c_X)/camera.c_X*180./pi

    info=BalisticInfo(u_0,fi_v,D)

    return info

def GetCoeff(D:float):# CoeffVx,CoeffVy
    return D*1.2, D*0.9    

def sign(x):
    if x>0: return 1
    if x==0: return 0
    if x<0: return -1

def GetV(D1:float,D2:float,f1:float,f2:float,u1:float,u2:float):
    Vx=sqrt(D1**2+D1**2-2*D1*D1*cos((f2-f1)*pi/180))*sign(f2-f1)
    Vy=sqrt(D1**2+D2**2-2*D1*D2*cos((u2-u1)*pi/180))*sign(u2-u1)
    return Vx,Vy

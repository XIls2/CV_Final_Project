import sys
import cv2
import math
import numpy as np

#暗通道
def Dark_Channel(I,sr):
    b,g,r = cv2.split(I)
    src = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2*sr+1,2*sr+1))
    dark = cv2.erode(src,kernel)
    return dark

#大气光强度A
def Atmospheric_Light(I,dark):
    [height,width] = I.shape[:2]
    I_size = height * width
    numpx = int(max(math.floor(I_size/1000),1))
    Dark_vec = dark.reshape(I_size,1)
    I_vec = I.reshape(I_size,3)
    indices = Dark_vec.argsort()
    indices = indices[I_size-numpx::]
    atmsum = np.zeros([1,3])
    for i in range(1,numpx):
       atmsum = atmsum + I_vec[indices[i]]
    A = atmsum / numpx
    return A

#透射率t(x)
def Estimate_Transmission(I,A,r,omega):
    I_norm = np.empty(I.shape,I.dtype)
    for ind in range(0,3):
        I_norm[:,:,ind] = I[:,:,ind]/A[0,ind]
    transmission = 1 - omega*Dark_Channel(I_norm,r)
    return transmission

#导向滤波
def GuideFilter(I,p,r,eps):
    mean_I = cv2.boxFilter(I,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(I*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p
    mean_II = cv2.boxFilter(I*I,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I
 
    a = cov_Ip/(var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))
    return mean_a * I + mean_b
 
def TransmissionRefine(I, Tm):
    gray = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    #GuideFilter中的均值滤波半径
    r = 60
    #确保分母不为0
    eps = 0.0001
    t = GuideFilter(gray,Tm,r,eps)
    return t
 
def Recover(I,t,A,tx):
    res = np.empty(I.shape,I.dtype)
    t = cv2.max(t,tx)
    for ind in range(0,3):
        res[:,:,ind] = (I[:,:,ind]-A[0,ind])/t + A[0,ind]
    return res
 
if __name__ == '__main__':
    fn = 'medium.jpg'
    src = cv2.imread(fn)
    I = src.astype('float64')/255

    #暗通道最小值滤波半径
    r=15
    #去除模糊的程度
    omega=0.95
    tx=0.1
    dark = Dark_Channel(I,r)
    cv2.imwrite("Dark_medium_r15.jpg",dark*255)
    A = Atmospheric_Light(I,dark)
    Tm = Estimate_Transmission(I,A,r,omega)
    t = TransmissionRefine(src,Tm)
    J = Recover(I,t,A,tx)
    
    arr = np.hstack((I, J))
    cv2.imwrite("Recover_medium_r15.jpg", J*255 )
    cv2.imwrite("Contrast_medium_r15.jpg", arr*255)
    cv2.waitKey()

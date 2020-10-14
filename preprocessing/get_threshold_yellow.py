import cv2
import numpy as np


def nothing(x):
    pass



def set_mean_brightness(img, rgb = [96,96,96]):
    m0 = rgb[0] - np.mean(img_[:,:,0])
    m1 = rgb[1] - np.mean(img_[:,:,1])
    m2 = rgb[2] - np.mean(img_[:,:,2])
    return cv2.add(img,np.ones(img_.shape, dtype=np.int8)*[m0,m1,m2], dtype=0)


img_ = cv2.imread('../imagenes/img_20.jpg',1)

h, w, d = img_.shape
img_ = cv2.resize(img_,(int(w/5), int(h/5)))

hsv = cv2.cvtColor(img_, cv2.COLOR_BGR2HSV)

lower = np.array([10, 120, 100])
upper = np.array([28, 255, 255])


mask = cv2.inRange(hsv, lower, upper)

cv2.imshow('screen', mask)

add = set_mean_brightness(img_)

cv2.namedWindow("Ventana Trackbar")
cv2.createTrackbar("umbral_azul","Ventana Trackbar",0, 255, nothing)
cv2.createTrackbar("umbral_rojo","Ventana Trackbar",0, 255, nothing)
cv2.createTrackbar("umbral_verde","Ventana Trackbar",0, 255, nothing)
cv2.setTrackbarPos('umbral_azul','Ventana Trackbar', 40)
cv2.setTrackbarPos('umbral_rojo','Ventana Trackbar',45)
cv2.setTrackbarPos('umbral_verde','Ventana Trackbar',35)

while (True):
    cv2.imshow("Ventana Trackbar", add)
    if ((cv2.waitKey(1) & 0xFF)==ord('q')):
        break
    U1=cv2.getTrackbarPos('umbral_azul','Ventana Trackbar')
    U2=cv2.getTrackbarPos('umbral_verde','Ventana Trackbar')
    U3=cv2.getTrackbarPos('umbral_rojo','Ventana Trackbar')

    ret, img_bin_b = cv2.threshold(add[:,:,0],U1,255,cv2.THRESH_BINARY)
    ret2, img_bin_g = cv2.threshold(add[:,:,1],U2,255,cv2.THRESH_BINARY)
    ret2, img_bin_r = cv2.threshold(add[:,:,2],U3,255,cv2.THRESH_BINARY)


    result = cv2.subtract(cv2.bitwise_and(img_bin_g,img_bin_r), img_bin_b)
    cv2.imshow('resultado', result)

cv2.destroyAllWindows();


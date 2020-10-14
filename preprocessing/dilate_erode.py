import cv2
import numpy as np

def nothing(x):
    pass


## Para estandarizar el brillo de la imagen
def set_mean_brightness(img, rgb = [96,96,96]):
    m0 = rgb[0] - np.mean(img_[:,:,0])
    m1 = rgb[1] - np.mean(img_[:,:,1])
    m2 = rgb[2] - np.mean(img_[:,:,2])
    return cv2.add(img,np.ones(img_.shape, dtype=np.int8)*[m0,m1,m2], dtype=0)

## Tamaño de la imagen
def set_size(img, high = 960):
    h, w, d = img.shape
    ratio = high/h
    return cv2.resize(img,(int(w*ratio), high))


## para obtener la binarizacion en un rango de colores
def get_mask_color(img, lower, upper):
    ## [10, 120, 100],[28, 255, 255] para el amarillo
    ##np.array([10, 150, 0]),np.array([28, 255, 150]) para el negro
    hsv = cv2.cvtColor(img_, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return mask

## dilatar y erosionar la imagen
def dilate_erode(img, k1, i1, k2, i2):
    kernel = np.ones(k1)
    ans = cv2.erode(img, kernel, iterations=i1)
    kernel = np.ones(k2)
    ans = cv2.dilate(ans, kernel, iterations=i2)
    return ans
def dilate_erode_2(img, k1, i1, k2, i2):
    kernel = np.ones(k1)
    ans = cv2.erode(img, kernel, iterations=i1)
    kernel = np.ones(k2)
    ans = cv2.dilate(ans, kernel, iterations=i2)
    ans = cv2.erode(ans, kernel, iterations=i2)
    kernel = np.ones(k1)
    ans = cv2.dilate(ans, kernel, iterations=i1-1)
    return ans


## La proporcion de puntos negros en un contorno amarillo
def get_countour_points_ratio(img, contour):
    mask = np.zeros(img.shape,dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    sum_1 = np.sum(mask)
    mask = cv2.bitwise_and(img, mask)
    sum_2 = np.sum(mask)
    return sum_2/sum_1

## Proporcion del area de un contorno respecto a su bounding box
def get_box_ratio(contour):
    x,y,w,h=cv2.boundingRect(cont)
    return cv2.contourArea(cont)/(h*w)


## aislar la placa y poner los otros pixeles en negro
def isolate(img, mask):
    return cv2.multiply(img, mask, dtype = 2)

## Se lee la imagen y se pone del tamaño apropiado
img_ = cv2.imread('../imagenes/img_4.jpg',1)
h, w, d = img_.shape
img_ = set_size(img_)
h, w, d = img_.shape
img = np.copy(img_)
img_ = set_mean_brightness(img_)

#Para binarizar la imagen con umbrales para obtener el color amarillo de la placa
img_bin_yellow = get_mask_color(img_,np.array([10, 90, 0]),np.array([28, 255, 255]))

#se detecta el color amarillo, se erosionan y dilata la imagen para obtener contornos mas definidos
result_yellow = dilate_erode_2(img_bin_yellow,(2,2),2,(5,5),4)

#Para binarizar la imagen con umbrales para obtener el color negro de la placa
img_bin_black = get_mask_color(img_,np.array([10, 120, 0]),np.array([28, 255, 150]))
#se detecta el color negro, se erosionan y dilata la imagen para obtener contornos mas definidos
result_black = dilate_erode(img_bin_black,(2,2),1,(4,4),4)


cv2.imshow('original', img)
#cv2.imshow('resultado_yellow', result_yellow)
#cv2.imshow('resultado_black', result_black)
cv2.waitKey(0)


#se obtienen los contornos y la bounding boxes
cnt, hie = cv2.findContours(result_yellow.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
for i, cont in enumerate(cnt):
    r1 = get_box_ratio(cont)
    r2 = get_countour_points_ratio(result_black, cont)
    x,y,w,h=cv2.boundingRect(cont)
    #print(r1,r2)
    u1 = 0.65
    u2 = 0.15
    u3 = 1200
    if (r1 > u1) and (r2 > u2) and (h*w > u3):
        #print('entro!')
        #cv2.rectangle(img,(x,y),(x+w, y+h), (0,255,0))
        arr = np.zeros((h,w,3))
        arr[:,:,0] = result_yellow[y:y+h,x:x+w]
        arr[:,:,1] = result_yellow[y:y+h,x:x+w]
        arr[:,:,2] = result_yellow[y:y+h,x:x+w]
        placa = isolate(img[y:y+h,x:x+w], arr)
        print(type(placa[0,0,0]))
        cv2.imshow('placa', placa)
        cv2.waitKey(0)
        cv2.imwrite('../outputs/placa.png', placa)


cv2.imshow('original', img)

cv2.waitKey(0)
cv2.destroyAllWindows();

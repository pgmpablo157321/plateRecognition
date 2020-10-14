import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib

class plateDetection():
    def __init__(self, path):
        self.img = cv2.imread(path,1)
    ## Para estandarizar el brillo de la imagen
    def set_mean_brightness(self, img, rgb = [96,96,96]):
        m0 = rgb[0] - np.mean(img[:,:,0])
        m1 = rgb[1] - np.mean(img[:,:,1])
        m2 = rgb[2] - np.mean(img[:,:,2])
        return cv2.add(img,np.ones(img.shape, dtype=np.int8)*[m0,m1,m2], dtype=0)

    ## Tamaño de la imagen
    def set_size(self, img, high = 960):
        h, w, d = img.shape
        ratio = high/h
        return cv2.resize(img,(int(w*ratio), high))


    ## para obtener la binarizacion en un rango de colores
    def get_mask_color(self, img, lower, upper):
        ## [10, 120, 100],[28, 255, 255] para el amarillo
        ##np.array([10, 150, 0]),np.array([28, 255, 150]) para el negro
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        return mask

    ## dilatar y erosionar la imagen
    def dilate_erode(self, img, k1, i1, k2, i2):
        kernel = np.ones(k1)
        ans = cv2.erode(img, kernel, iterations=i1)
        kernel = np.ones(k2)
        ans = cv2.dilate(ans, kernel, iterations=i2)
        return ans
    def dilate_erode_2(self, img, k1, i1, k2, i2):
        kernel = np.ones(k1)
        ans = cv2.erode(img, kernel, iterations=i1)
        kernel = np.ones(k2)
        ans = cv2.dilate(ans, kernel, iterations=i2)
        ans = cv2.erode(ans, kernel, iterations=i2)
        kernel = np.ones(k1)
        ans = cv2.dilate(ans, kernel, iterations=i1-1)
        return ans


    ## La proporcion de puntos negros en un contorno amarillo
    def get_countour_points_ratio(self, img, contour):
        mask = np.zeros(img.shape,dtype="uint8")
        cv2.drawContours(mask, [contour], -1, 255, -1)
        sum_1 = np.sum(mask)
        mask = cv2.bitwise_and(img, mask)
        sum_2 = np.sum(mask)
        return sum_2/sum_1

    ## Proporcion del area de un contorno respecto a su bounding box
    def get_box_ratio(self, contour):
        x,y,w,h=cv2.boundingRect(contour)
        return cv2.contourArea(contour)/(h*w)


    ## aislar la placa y poner los otros pixeles en negro
    def isolate(self, img, mask):
        return cv2.multiply(img, mask, dtype = 2)


    def preprocess(self, img = None):
        if img is None:
            img = self.img
        img_ = self.set_size(img)
        h, w, d = img_.shape
        img_ = self.set_mean_brightness(img_)
        
        #Se binariza detectando el color amarillo, se dilata y se erosiona
        img_bin_yellow = self.get_mask_color(img_,np.array([10, 90, 0]),np.array([28, 255, 255]))
        result_yellow = self.dilate_erode_2(img_bin_yellow,(2,2),2,(5,5),4)
        #Se binariza detectando el color negro, se dilata y se erosiona
        img_bin_black = self.get_mask_color(img_,np.array([10, 120, 0]),np.array([28, 255, 150]))
        result_black = self.dilate_erode(img_bin_black,(2,2),1,(4,4),4)

        #se obtienen los contornos y la bounding boxes
        cnt, hie = cv2.findContours(result_yellow.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        
        array_placas = []
        for i, cont in enumerate(cnt):
            r1 = self.get_box_ratio(cont)
            r2 = self.get_countour_points_ratio(result_black, cont)
            x,y,w,h=cv2.boundingRect(cont)
            u1 = 0.8
            u2 = 0.15
            u3 = 1200
            if (r1 > u1) and (r2 > u2) and (h*w > u3):
                arr = np.zeros((h,w,3))
                arr[:,:,0] = result_yellow[y:y+h,x:x+w]
                arr[:,:,1] = result_yellow[y:y+h,x:x+w]
                arr[:,:,2] = result_yellow[y:y+h,x:x+w]
                placa = self.isolate(img_[y:y+h,x:x+w], arr)
                array_placas.append(placa.copy())
        return array_placas



class plateDivision():
    def dilate_erode(self, img, k1, i1, k2, i2):
        kernel = np.ones(k1)
        ans = cv2.erode(img, kernel, iterations=i1)
        kernel = np.ones(k2)
        ans = cv2.dilate(ans, kernel, iterations=i2)
        return ans

    def erode_dilate(self, img, k1, i1, k2, i2):
        kernel = np.ones(k2)
        ans = cv2.dilate(img, kernel, iterations=i2)
        kernel = np.ones(k1)
        ans = cv2.erode(ans, kernel, iterations=i1)
        return ans
    def preprocess(self, img):
        h, w, d = img.shape
        copy = np.reshape(img, (h*w, d))
        km = KMeans(n_clusters=2)
        y_pred = km.fit_predict(copy)
        if ((y_pred==0).sum()<(y_pred==1).sum()):
            y_pred = 1-y_pred

        arr = np.zeros((h,w,d), dtype=np.uint8)
        arr[:,:,0] = y_pred.reshape((h,w))*255
        arr[:,:,1] = y_pred.reshape((h,w))*255
        arr[:,:,2] = y_pred.reshape((h,w))*255

        arr = self.dilate_erode(arr, (2,1),1,(2,1),1)
        #arr = self.erode_dilate(arr, (1,2),1,(1,2),1)
        img_cont = arr[:,:,0].astype(np.uint8)
        cnt, hie = cv2.findContours(img_cont, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        character_array = []

        for i, cont in enumerate(cnt):
            x,y,wc,hc=cv2.boundingRect(cont)
            u1 = (w*h)/25
            u2 = (w/12)
            u3 = (h/3)
            u4 = (w*h/2)
            if (wc*hc > u1) and (wc>u2) and (hc>u3) and (wc*hc<u4):
                #print(arr[y-1:y+hc+1,x-1:x+wc+1, 0])
                character = cv2.resize(arr[y:y+hc,x-1:x+wc+1, 0],(15,30))
                character_array.append((x, character.copy()))
                #cv2.imshow('digito', cv2.resize(img_[y-1:y+hc+1,x-1:x+wc+1],(15,30)))
                #cv2.imshow('digito_bin', cv2.resize(character,(15,30)))
                #cv2.waitKey(0)

        character_array.sort(key=lambda x: x[0])
        return character_array


class digitClasification():
    def __init__(self):
        self.nn_numeros = joblib.load('../model/saved/nn_numbers.joblib')
        self.nn_letras = joblib.load('../model/saved/nn_letters.joblib')
        self.sc_numeros = joblib.load('../model/saved/scaler_numbers.joblib')
        self.sc_letras = joblib.load('../model/saved/scaler_letters.joblib')
        self.letter_dig = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',
                18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23: 'X', 24:'Y', 25:'Z'}
    def predict(self, x, pos):
        if pos < 3:
            ans = self.sc_letras.transform(x)
            ans = self.nn_letras.predict(ans)
            ans = self.letter_dig[ans[0]]
        else:
            ans = self.sc_numeros.transform(x)
            ans = self.nn_numeros.predict(ans)
            ans = int(ans[0])
        return ans




pDet = plateDetection('../imagenes/img_10.jpg')
pDiv = plateDivision()
dClas = digitClasification()
arr = pDet.preprocess()

for e in arr:
    cv2.imshow('plate',e)
    cv2.waitKey(0)
    for i, e1 in enumerate(pDiv.preprocess(e)):
        cv2.imshow('digit',e1[1])
        #print(e1[1])
        print(dClas.predict(e1[1].reshape(1,-1), i), end='')
        cv2.waitKey(0)
    print()
cv2.destroyAllWindows()

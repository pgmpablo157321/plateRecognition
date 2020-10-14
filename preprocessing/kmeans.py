import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib


def dilate_erode(img, k1, i1, k2, i2):
    kernel = np.ones(k1)
    ans = cv2.erode(img, kernel, iterations=i1)
    kernel = np.ones(k2)
    ans = cv2.dilate(ans, kernel, iterations=i2)
    return ans


def erode_dilate(img, k1, i1, k2, i2):
    kernel = np.ones(k2)
    ans = cv2.dilate(img, kernel, iterations=i2)
    kernel = np.ones(k1)
    ans = cv2.erode(ans, kernel, iterations=i1)
    return ans



img_ = cv2.imread('../outputs/placa.png',1)
h, w, d = img_.shape
nn_numeros = joblib.load('../model/saved/nn_numbers.joblib')
nn_letras = joblib.load('../model/saved/nn_letters.joblib')
sc_numeros = joblib.load('../model/saved/scaler_numbers.joblib')
sc_letras = joblib.load('../model/saved/scaler_letters.joblib')
letter_dig = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',
                18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23: 'X', 24:'Y', 25:'Z'}


copy = np.reshape(img_, (h*w, d))

km = KMeans(n_clusters=2)
y_pred = km.fit_predict(copy)
if ((y_pred==0).sum()<(y_pred==1).sum()):
    y_pred = 1-y_pred


arr = np.zeros((h,w,d), dtype=np.uint8)
arr[:,:,0] = y_pred.reshape((h,w))*255
arr[:,:,1] = y_pred.reshape((h,w))*255
arr[:,:,2] = y_pred.reshape((h,w))*255

arr = dilate_erode(arr, (2,1),1,(2,1),1)
#arr = erode_dilate(arr, (1,2),1,(1,2),1)

img_cont = arr[:,:,0].astype(np.uint8)
#print(img_cont, type(img_cont[0,0]))
cv2.imshow('cluster',arr)
cv2.waitKey(0)
cnt, hie = cv2.findContours(img_cont, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

character_array = []

for i, cont in enumerate(cnt):
    x,y,wc,hc=cv2.boundingRect(cont)
    u1 = (w*h)/25
    u2 = (w/12)
    u3 = (h/3)
    u4 = (w*h/2)
    if (wc*hc > u1) and (wc>u2) and (hc>u3) and (wc*hc<u4):
        character = cv2.resize(arr[y-1:y+hc+1,x-1:x+wc+1, 0],(15,30))
        character_array.append((x, character.copy()))
        #cv2.imshow('digito', cv2.resize(img_[y-1:y+hc+1,x-1:x+wc+1],(15,30)))
        #cv2.imshow('digito_bin', cv2.resize(character,(15,30)))
        #cv2.waitKey(0)

character_array.sort(key=lambda x: x[0])

for i, e in enumerate(character_array):
    cv2.imshow('digito_bin', e[1])
    dig = e[1].copy()
    dig = dig.reshape(1,-1)
    
    if i < 3:
        scaled_dig = sc_letras.transform(dig)
        prediction = nn_letras.predict(scaled_dig)
        prediction = letter_dig[int(prediction[0])]
    else:
        scaled_dig = sc_numeros.transform(dig)
        prediction = nn_numeros.predict(scaled_dig)
        prediction = int(prediction[0])
    print(prediction, end='')
    cv2.waitKey(0)

#cv2.imshow('placa', img_)
cv2.destroyAllWindows()
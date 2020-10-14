import numpy as np
import pandas as pd
import cv2
import math

from os import listdir
from os.path import isfile, join


def set_size(img, high = 30):
    h, w, d = img.shape
    ratio = high/h
    return cv2.resize(img,(int(w*ratio), high))

total_samples = 7000
n_samples = 2000


vectorFolders = ["A", "B", "C", "D", "E","F", "G", "H", "I", "J", "K", "L", "M", "N", "O","P", "Q", "R", "S", "T","U","V","W","X","Y","Z"]
mypath = "letter_data/"
files = []
for value in vectorFolders:
    files.extend([mypath+value+"/"+f for f in listdir(mypath+value+"/")[:total_samples] if isfile(join(mypath+value+"/", f))])
    print(len(files))

data = np.zeros((n_samples*len(vectorFolders), 451), dtype = np.uint8)
data_index=0

for i in range(len(vectorFolders)):
    j=0
    index = 0
    print(i)
    while (j<n_samples and index < total_samples):
        img = cv2.imread(files[i*total_samples+index])
        img = set_size(img)
        w = img.shape[1]
        if w <= 15:
            i1 = math.floor((15-w)/2)
            i2 = i1+w
            blanck = np.zeros((30,15), dtype=np.uint8)
            blanck[:,i1:i2] = img[:,:,0]
            data[data_index, 1:]=blanck.reshape(1,-1)
            data[data_index,0]=i
            data_index+=1
            j+=1
        index+=1


np.savetxt('extracted_data/data_letters.csv', data)






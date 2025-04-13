import cv2
import numpy as np
from matplotlib import pyplot as plt

buyuk_resim=cv2.imread('ev.jpg')

print(str(len(buyuk_resim.shape)))

print(buyuk_resim.dtype)

cv2.imshow('birinci',buyuk_resim)
kucuk_resim=cv2.imread('ucak.png')
cv2.imshow('ikinci',kucuk_resim)

satir,sutun,kanal=kucuk_resim.shape
print(satir,sutun,kanal)
kucuk_resim_boyut


kucuk_resim_gri=cv2.cvtColor(kucuk_resim,cv2.COLOR_BGR2GRAY)
cv2.waitKey(0)

dene=cv2.resize(kucuk_resim_gri,(400,400))
cv2.imshow('bigger size',dene)



ret, mask=cv2.threshold(kucuk_resim_gri,0,255,cv2.THRESH_BINARY)

dene1=cv2.resize(ret,(400,400))
cv2.imshow('biggersize ret',dene1)

dene2=cv2.resize
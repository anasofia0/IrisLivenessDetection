import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def trata_img(img):

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(11,11))

    blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

    img = img+blackhat

    img = cv.GaussianBlur(img, (31,31), 1)
    # cv.imshow('', img)
    # cv.waitKey(0)

    return img

def segmenta_iris(img):

    pupila = cv.Canny(img,50,100)
    iris = cv.Canny(img,20,20)

    circulos = cv.HoughCircles(pupila,cv.HOUGH_GRADIENT,1,20,
                                param1=50,param2=30,minRadius=0,maxRadius=0)

    c = np.uint16(np.around(circulos))
    c = (c[0,0,0], c[0,0,1], c[0,0,2])

    circulos = cv.HoughCircles(iris,cv.HOUGH_GRADIENT,1,20,
                                param1=50,param2=30,minRadius=0,maxRadius=0)

    raios = np.uint16(np.around(circulos))
    raios= raios[0,:]

    raios = np.uint16(np.around(raios))
    raios=raios[:,2]

    cond = np.logical_and(raios>90, raios<120)
    raio = int(np.mean(raios[np.where(cond)]))

    return c, raio

def cria_mascara(img, pupila, r_iris):

    print(img.shape)

    mask = np.zeros(img.shape)
    mask_aux = np.zeros(img.shape)

    branco = (255,255,255)

    mask = cv.circle(mask, (pupila[0],pupila[1]), r_iris, branco, -1)
    mask_aux = cv.circle(mask_aux, (pupila[0],pupila[1]), pupila[2], branco, -1)

    mask = cv.subtract(mask, mask_aux)
    mask = mask.astype('uint8')

    mask = cv.bitwise_and(mask, img)

    return mask

def filtro_res1(img):

    row, col = img.shape

    x1 = np.copy(img)
    x1 = x1.astype('uint64')
    x2 = np.copy(img)
    x2 = x2.astype('uint64')

    for i in range(row):
        for j in range(col-1):
            
            x1[i,j] = x1[i,j+1] - x1[i,j]

    for i in range(row-1):
        for j in range(col):
            x2[i,j] = x2[i+1,j] - x2[i,j]

    r = np.minimum(x1, x2)

    return r.astype('uint8')

"""
MAIN
"""

img = cv.imread('database/CASIA1/17/017_1_1.jpg', 0)
# img = cv.imread('database/CASIA-Iris-Interval/003/R/S1003R04.jpg', 0)

cv.imshow('', img)
cv.waitKey(0)

img_trat = trata_img(img)
c, raio = segmenta_iris(img_trat)

img_seg = np.copy(img)

cv.circle(img_seg,(c[0],c[1]),c[2],(255,0,0),2)
cv.circle(img_seg,(c[0],c[1]),int(raio*1.01),(255,0,0),2)

cv.imshow('', img_seg)
cv.waitKey(0)

mascara = cria_mascara(img, c, raio)
filtro = filtro_res1(mascara)

cv.imshow('', filtro)
cv.waitKey(0)
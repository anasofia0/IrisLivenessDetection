import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd

def trata_img(img):

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(11,11))

    blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

    img = img+blackhat

    img = cv.GaussianBlur(img, (35,35), 1)
    # cv.imshow('', img)
    # cv.waitKey(0)

    return img

def segmenta_iris(img):

    pupila = cv.Canny(img,60,100)
    iris = cv.Canny(img,20,20)

    cv.imshow('', pupila)
    cv.waitKey(0)

    cv.imshow('', iris)
    cv.waitKey(0)
    

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

def feature_extraction(path):

    files = os.listdir(path)

    star = cv.xfeatures2d.StarDetector_create()
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

    df = pd.DataFrame(columns=['class', 'desc'])
    df_aux = pd.DataFrame(index=range(len(files)) ,columns=range(256))

    d = []
    c = []

    for i in files:

        img = cv.imread(path+i)

        kp = star.detect(img,None)
        kp, des = brief.compute(img, kp)

        vf = 0 if i[-5] == 'f' else 1

        c.append(vf)
        if des is not None:
            d.append(des)
        else:
            d.append([])

    df['class'] = c
    df['desc'] = d

    hist = []

    for i in d:
        hist.append(np.histogram(i, bins=256)[0])

    print(hist)

    df_aux[:] = hist

    return df, df_aux

def model_training(df, hist):

    param_grid={'C':[0.1,1,10,100],
                'gamma':[0.0001,0.001,0.1,1],
                'kernel':['rbf','poly']}

    svc=svm.SVC(probability=True)
    model=GridSearchCV(svc,param_grid)

    model.fit(hist.iloc[:30], df.iloc[:30,0])

    # testando

    pred = model.predict(hist.iloc[30:])
    esperado = df.iloc[30:,0]

    print(f'acuracia: {accuracy_score(esperado,pred)*100}%')
    
def pega_img():
    
    lista = os.listdir('/home/anasofia/Downloads/OLHO/')

    c = 1
    for i in lista:
        os.system(f'cp /home/anasofia/Downloads/OLHO/{i} database/olhos/falso/')
        os.system(f'mv database/olhos/falso/{i} database/olhos/falso/{c}.jpg')
        c += 1

def cria_database(quant):

    for i in range(quant):
        os.system(f'mv database/olhos/verdadeiro/{(f"{i+1}").zfill(3)}_1_1.jpg database/olhos/verdadeiro/{i+1}.jpg')
        
def pre_processing():

    for i in range(20):

        img = cv.imread(f'database/olhos/verdadeiro/{i+1}.jpg', 0)

        img_trat = trata_img(img)
        c, raio = segmenta_iris(img_trat)

        # img_seg = np.copy(img)

        # cv.circle(img_seg,(c[0],c[1]),c[2],(255,0,0),2)
        # cv.circle(img_seg,(c[0],c[1]),int(raio*1.01),(255,0,0),2)

        mascara = cria_mascara(img, c, raio)
        cv.imwrite(f'database/olhos/tratado/{i+1}_v.jpg', mascara)
        filtro = filtro_res1(mascara)
        cv.imwrite(f'database/olhos/tratado_filtro/{i+1}_v.jpg', filtro)
    
    for i in range(20):

        img = cv.imread(f'database/olhos/falso/{i+1}.jpg', 0)

        img = img[:, 1000:5700]
        img = cv.resize(img, (320, 280), cv.INTER_CUBIC)

        img_trat = trata_img(img)
        c, raio = segmenta_iris(img_trat)

        # img_seg = np.copy(img)

        # cv.circle(img_seg,(c[0],c[1]),c[2],(255,0,0),2)
        # cv.circle(img_seg,(c[0],c[1]),int(raio*1.01),(255,0,0),2)

        mascara = cria_mascara(img, c, raio)
        cv.imwrite(f'database/olhos/tratado/{i+1}_f.jpg', mascara)
        filtro = filtro_res1(mascara)
        cv.imwrite(f'database/olhos/tratado_filtro/{i+1}_f.jpg', filtro)

"""
MAIN
"""

# print('treinando com os dados sem filtro:', end='')
df, hist = feature_extraction('database/olhos/tratado/')

hist = np.histogram(df.iloc[0,1], bins=256)[0]

plt.hist(hist, bins=256)
plt.show()

# model_training(df, hist)

# print('treinando com os dados com filtro:', end='')
# df, hist = feature_extraction('database/olhos/tratado_filtro/')
# model_training(df, hist)

img = cv.imread('database/olhos/verdadeiro/1.jpg')

cv.imshow('', img)
cv.waitKey(0)

trat = trata_img(img)
c, r = segmenta_iris(trat)
masc = cria_mascara(img, c, r)

cv.imshow('', masc)
cv.waitKey(0)

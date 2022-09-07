import cv2 as cv
import numpy as np

def segmenta_iris(img):

    # suavizada = cv.medianBlur(img,7)
    # _,limites = cv.threshold(suavizada,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    # bordas = cv.Canny(limites,0,0)

    # cv.imshow('',bordas)
    # cv.waitKey(0)

    # img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # img_gray = cv.GaussianBlur(img_gray, (31, 31), 1)

    img_gray = cv.medianBlur(img, 11)

    # kernel = np.array([[-1,-1,-1],
    #                     [-1,9,-1],
    #                     [-1,-1,-1]])

    # img_gray = cv.filter2D(img_gray,-1,kernel)

    img_gray = cv.threshold(img_gray,140,255,cv.THRESH_BINARY)[1]

    contornos = cv.findContours(img_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

    img_contorno = np.copy(img_gray)
    cv.drawContours(img_contorno, contornos, -1, (0, 255, 0))

    cv.imshow('aaaaaaaaaaaaaaaa', img_contorno)
    cv.waitKey(0)

    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(11,11))
    # img_gray = cv.morphologyEx(img_gray, cv.MORPH_CLOSE, kernel, iterations=2)

    # img_gray = cv.Canny(img_gray, 50, 50)

    # cv.imshow('', img_gray)
    # cv.waitKey(0)

    # cv.imshow('', img_gray)
    # cv.waitKey(0)

    circulos = cv.HoughCircles(img_contorno, cv.HOUGH_GRADIENT, 1,20,
                                param1=150,param2=30,minRadius=0, maxRadius=500)

    if circulos is not None:
        circulos = np.uint16(np.around(circulos))
        for i in circulos[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(img, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(img, center, radius, (255, 0, 255), 3)

    cv.imshow("detected circles", img)
    cv.waitKey(0)

def filtro_res1(img):

    row, col = img.shape

    x1 = np.copy(img)
    x2 = np.copy(img)

    for i in range(row):
        for j in range(col-1):
            x1[i,j] = x1[i,j+1] - x1[i,j]

    for i in range(row-1):
        for j in range(col):
            x2[i,j] = x2[i+1,j] - x2[i,j]

    r = np.minimum(x1, x2)

    return r

def filtro_res2(img):

    row, col = img.shape
    x = np.copy(img)

    for i in range(row):
        for j in range(1,col-1):
            
            x[i,j] = x[i,j-1] + x[i,j+1] - 2*x[i,j]
    
    return x

def aplica_filtros(img):

    pass

img = cv.imread('database/olho.jpg', 0)
# img = cv.imread('database/iris57_64/057L_1.png', 0)
cv.imshow('', img)
cv.waitKey(0)

img = filtro_res2(img)

cv.imshow('', img)
cv.waitKey(0)
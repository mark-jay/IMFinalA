import cv2
import numpy as np
from numpy import array

# image source
allImages = ["P1000697s.jpg", "P1000698s.jpg", "P1000697s.jpg", 
            "P1000699s.jpg", "P1000703s.jpg", "P1000705s.jpg", 
            "P1000706s.jpg", "P1000709s.jpg", "P1000710s.jpg", "P1000713s.jpg"]

def myRead():
    # import sys
    # sys.path.append('G:\sting\univer\master 2\IMAGE PROCESSING\Lab 1\images')    
    return map(cv2.imread, allImages)

def myRead1(idx = 0):
    return map(cv2.imread, [allImages[idx]])

#another way: im_gray = cv2.imread('grayscale_image.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
def mycvtConvert(im):
    return map(lambda x:cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), im)

def myPrint(im):
    winName = '__'
    cv2.namedWindow(winName)
    for x in range(0,len(im)):
        cv2.imshow(winName, im[x])
        cv2.waitKey(0)
    cv2.destroyWindow(winName)

def myThreshold(ims):
    def f(i):
        #thresh = 110
        #im_bw = cv2.threshold(im[x], thresh, 255, cv2.THRESH_BINARY)[1]
        (treshold, _) = cv2.threshold(i, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return cv2.threshold(i, treshold-10, 255, cv2.THRESH_BINARY)[1]
    return map(f, ims)

def getKernel(n = 7):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n,n))

def myDilate(im, kernel = getKernel()):
    return map(lambda i: cv2.dilate(i,kernel), im)

def myContours1(im):
    def f(i):
        (contours , hierarchy1)= cv2.findContours(np.array(i), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow(winName, contours[x])
        # myPrint(contours)
        # cv2.drawContours(i, contours, -1, 150, hierarchy = hierarchy1)
        cv2.drawContours(i, contours, 0, 255, 2, 
                                hierarchy = hierarchy1)
        return i
    return map(f,im)

def closeMO(im, kernel = getKernel()):
    return map(lambda i: cv2.erode(cv2.dilate(i,kernel), kernel), im)    

def openMO(im, kernel = getKernel()):
    return map(lambda i: cv2.dilate(cv2.erode(i,kernel), kernel), im)    

def myContours(im):
    def f(i):
        i = np.array(i)
        (contours , hierarchy1)= cv2.findContours(i, 1 , 1)
        # cv2.imshow(winName, contours[x])
        # myPrint(contours)
        cv2.drawContours(i, contours, -1, 150,  hierarchy = hierarchy1)
        return i
    return map(f,im)

def run():
    im1 = myRead()
    im2 = mycvtConvert(im1)
    im3 = myThreshold(im2)
    # im4 = myContours(im3)
    
    #print (im3 == im4) # error - why?
    #myPrint(im3)
    
    # myPrint(closeMO(im4, getKernel(10)))
    im4 = closeMO(im3)
    im5 = openMO(im3, getKernel(2))
    myPrint(im5)
    # myPrint(im4)

run()

"""
im1 = myRead1()
im2 = mycvtConvert(im1)
im3 = myThreshold(im2)
im4 = myContours(im3)


myPrint(im4)

----------------------------------------------------------

def getKernel():
    #variant 0
   # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    #variant 1
    #y,x = np.ogrid[-3: 3+1, -3: 3+1]
    #mask = x**2+y**2 <= 3**2
    #kernel= array(map(lambda b:map(lambda a: (1 if a else 0), b), mask))   
    #variant 2
    #kernel = np.ones((7,7),'int')
    #variant 3    
    #kernel = np.array([[0,0,0,1,0,0,0],[0,1,1,1,1,1,0],[0,1,1,1,1,1,0],[1,1,1,1,1,1,1],[0,1,1,1,1,1,0],[0,1,1,1,1,1,0],[0,0,0,1,0,0,0]],'int')
    #variant 4
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    return kernel

def myBI2(im):
#    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
#    res = cv2.morphologyEx(im,cv2.MORPH_OPEN,kernel)
    des = cv2.bitwise_not(getKernel())
    contour,hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(des,[cnt],0,255,-1)
    im = cv2.bitwise_not(des)
    return im

"""

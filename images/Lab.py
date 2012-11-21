import cv2
import numpy as np
from numpy import array

# import sys
# sys.path.append('G:\sting\univer\master 2\IMAGE PROCESSING\Lab 1\images')    

# fs - listof functions
def comp(fs):
    def f(f1, f2):
        return lambda x: f1(f2(x))
    return reduce(f, fs)

# image source
allImages = ["P1000697s.jpg", "P1000698s.jpg", "P1000697s.jpg", 
            "P1000699s.jpg", "P1000703s.jpg", "P1000705s.jpg", 
            "P1000706s.jpg", "P1000709s.jpg", "P1000710s.jpg", "P1000713s.jpg"]

#another way: im_gray = cv2.imread('grayscale_image.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
def mycvtConvert(color = cv2.COLOR_BGR2GRAY):
    return lambda im : cv2.cvtColor(im, color)

def myPrint(im):
    winName = '__'
    cv2.namedWindow(winName)
    cv2.imshow(winName, im)
    cv2.waitKey(0)
    cv2.destroyWindow(winName)

def myThreshold(ims):
    (treshold, _) = cv2.threshold(ims, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return cv2.threshold(ims, treshold, 255, cv2.THRESH_BINARY)[1]

def getKernel(n = 7):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n,n))

def myContours1(im):
    im = np.array(im)
    (contours , hierarchy1)= cv2.findContours(np.array(im), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow(winName, contours[x])
    # myPrint(contours)
    # cv2.drawContours(i, contours, -1, 150, hierarchy = hierarchy1)
    cv2.drawContours(im, contours, 0, 255, 2, 
                            hierarchy = hierarchy1)
    return im

def dilate(kernel = getKernel()):
    return lambda img : cv2.dilate(img, kernel)

def erode(kernel = getKernel()):
    return lambda img : cv2.erode(img, kernel)

def closeMO(kernel = getKernel(), iterations = 1):
    return lambda im: cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel, 
                                       iterations = iterations)
    #cv2.erode(cv2.dilate(im, kernel), kernel)

def openMO(kernel = getKernel(), iterations = 1):
    return lambda im: cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel, 
                                       iterations = iterations)

def myContours(im):
    im = np.array(im)
    (contours , hierarchy1)= cv2.findContours(im, 1 , 1)
    # cv2.imshow(winName, contours[x])
    # myPrint(contours)
    cv2.drawContours(im, contours, -1, 150,  hierarchy = hierarchy1)
    return im
    
# n must be 0, 1 or 2
def splitFn(n): 
    def f(im):
        return cv2.split(im)[n]
    return f

# has a side effect: write the answer to the second arr
def sumMask(m1, m2):
    """
    for i in range(len(m1)):
        for j in range(len(m1[i])):
            # if (m1[i][j] > 0) | (m2[i][j] > 0):
                # m2[i][j] = 255
            # m1[i][j] = 100
            if (m1[i][j] > 0):
                pass
                #m2[i][j] = 255
    """
    return m1 + m2

def mkMaskCombinator(mFn1, mFn2):
    return lambda im : sumMask(mFn1(im), mFn2(im))

# 8, 5, 8, 5, 4, 4, 5, 3, 3, 3, 
def run():
    n = 10
    def printImg(f, imgs):
        map(comp([myPrint, f, cv2.imread]), imgs)

    generalContoursFn = comp([myContours, myThreshold, mycvtConvert()])
    defaultMaskFn = comp([closeMO(getKernel(n)), myThreshold, mycvtConvert()])
    # redMaskFn = comp([myPrint, closeMO(), myThreshold, mycvtConvert(), cv2.imread])
    redMaskFn = comp([closeMO(getKernel(n)), myThreshold, splitFn(2)]) # 2 is red
    
    maskFn = comp([erode(), erode(), erode(), mkMaskCombinator(defaultMaskFn, redMaskFn)])
    
    printImg(lambda x:x, allImages[:])
    printImg(maskFn, allImages[:])
    
    # printImg(redMaskFn, allImages[3:4])
    
    # printImg(generalContoursFn, allImages[:1])
    # printImg(maskFn, allImages[3:4])
    # printImg(redMaskFn, allImages[3:4])
    
    # printImg(generalFn, allImages[:1])


run()

import cv2
import numpy as np
from numpy import array
from copy import copy, deepcopy

""" hier [[[next, previous, firstChild, parent]]] """
""" getXY = lambda p : (p[0][0], p[0][1]) # kinda weird """


""" ----------------  combinator utils: printing """

""" a wrapper around a combinator 'f' and array of images. Then read, 
    apply function 'f' to the read image and print the result to a named window """
def printImg(f, imgs):
    map(comp([myPrint, f, cv2.imread]), imgs)

def printArr(img):
    print "printing arr: "
    print img
    return img

def printMaxMinVals(im):
    maxV = max(map(max,im))
    minV = min(map(min,im))
    print "(max = %s, min = %s) " % (maxV, minV)
    return im

def printTypes(im):
    print "im type = %s" % type(im)
    print "im[0] type = %s" % type(im[0])
    print "im[0][0] type = %s" % type(im[0][0])
    return im

def printAllValues(im):
    a1 = [v for a1 in im for v in a1]
    l = float(len(a1))
    m = {}
    def f(k):
        if k in m:
            m[k] = m[k] + 1
        else:
            m[k] = 1
    map(f, a1)
    probValues = map(lambda v : float(str(v / l)[:5]), m.values())
    print "all values: %s" % zip(m.keys(), probValues)
    return im

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

def mkThresholdFn(tr = 0):
    def tresholdFn(ims):
        if (tr == 0):
            return cv2.threshold(ims, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #(treshold, _) = cv2.threshold(ims, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return cv2.threshold(ims, tr, 255, cv2.THRESH_BINARY)[1]
    return tresholdFn

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

""" ----------------------- """
""" masks combinators """

def sumMasks(m1, m2):
    return m1 + m2

def combineMasks(combinator, mFn1, mFn2, type1 = int, type2 = np.float64):
    def f(im): 
        im1, im2 = np.array(mFn1(im), type1), np.array(mFn2(im), type1)
        return np.array(combinator(im1, im2), type2)
    return f




""" finds all the children of the holes. I.e. finds all inner holes """
def childrenIdxs(hier, firstChildIdx):
    idx = hier[0][firstChildIdx][2]
    childrenIdxs = [idx]
    while(hier[0][idx][0] != -1):
        idx = hier[0][idx][0] # first child
        childrenIdxs.append(idx)
    return childrenIdxs

""" counting sum of all the inner holes of the given hole """
def sumHolesArea(hier, contours, firstChildIdx):
    idxs = childrenIdxs(hier, firstChildIdx)
    return sum( map(lambda i : cv2.contourArea( contours[i] ), idxs) )

""" finds all the contours which have an area more than 'minArea', but < 'maxArea'
    returns all the contours and a list of the indexes that meet the 
    requirements """
def findAllContourByHolesArea(gray, minArea = 1700, maxArea = 1000000000):    
    contour,hier = cv2.findContours(gray,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
 
    idxs = []   
    for i, cnt in enumerate (contour):
        area = sumHolesArea (hier, contour, i)
        if (hier[0][i][2] != -1) & (area > minArea) & (area < maxArea):
            print "area = ", area
            idxs.append(i)
    
    return (contour, idxs)


def itemsWithBigHoles(orig):
    # 1500, 2100 min and max sizes of the holes of the red stuff
    (contour, idxs) = findAllContourByHolesArea(deepcopy(orig), 1800, 2100)
    gray = np.zeros((len (orig), len (orig[0])))
    
    cntIdx = 0
    color = 1
    thickness = -1 # Thickness of lines the contours are drawn with. If it is negative (for example, thickness=CV_FILLED ), the contour interiors are drawn.
    map(lambda i : cv2.drawContours(gray, [contour[i]], cntIdx, color, thickness),
        idxs)

    return gray

def fillSmallHoles(orig):
    (contour, idxs) = findAllContourByHolesArea(deepcopy(orig), 0, 500)
    gray = np.zeros((len (orig), len (orig[0])))
    
    cntIdx = 0
    color = 1
    thickness = -1 # Thickness of lines the contours are drawn with. If it is negative (for example, thickness=CV_FILLED ), the contour interiors are drawn.
    map(lambda i : cv2.drawContours(gray, [contour[i]], cntIdx, color, thickness),
        idxs)

    return gray

def invert(img):
    # return (img+255)%510
    return (img+1)%2


# 8, 5, 8, 5, 4, 4, 5, 3, 3, 3, 
# http://stackoverflow.com/questions/10316057/filling-holes-inside-a-binary-object

def run():
    n = 10
    def printImg(f, imgs):
        map(comp([myPrint, f, cv2.imread]), imgs)

    """ making a mask for coins """
    
    """ for all the coins except red(brown?) one,
        both close and open operations was not able to do what I wanted """
    defaultMaskFn = comp([fillSmallHoles, mkThresholdFn(), 
                          mycvtConvert()])
    """ 2 is a red color. for getting red coins """
    redMaskFn = comp([invert, 
                      itemsWithBigHoles, closeMO(getKernel(5)), 
                             mkThresholdFn(180), splitFn(2)])
    """ combination of both """
    maskFn = comp([combineMasks(cv2.bitwise_and, defaultMaskFn, redMaskFn)])
    #maskFn = comp([erode(), erode(), erode(), mkMaskCombinator(defaultMaskFn, redMaskFn)])
    
    # idk why mb will be useful later
    generalContoursFn = comp([myContours, mkThresholdFn(), mycvtConvert()])

    # printImg(lambda x:x, allImages[:])
    # printImg(maskFn, allImages[:])

    startN = 0
    lastN = 20 
    
    printImg(lambda x:x, allImages[startN:lastN])
    printImg(redMaskFn, allImages[startN:lastN])
    #printImg(defaultMaskFn, allImages[startN:lastN])
    #printImg(maskFn, allImages[startN:lastN])
    
    # printImg(someFun1, allImages[8:])
    # printImg(redMaskFn, allImages[3:4])
     
    # printImg(generalContoursFn, allImages[:])
    # printImg(maskFn, allImages[3:4])
    # printImg(redMaskFn, allImages[3:4])
    
    # printImg(generalFn, allImages[:1])


run()
import cv2
import numpy as np
from numpy import array
from copy import copy, deepcopy
from math import pi, sqrt
from operator import itemgetter, mul, add
from functools import partial

""" ----------------  utils, and local constants """

def getFeatures(contour):
    m = cv2.moments(contour)
    f = {}
    
    f['area'] = m['m00']
    f['perimeter'] = cv2.arcLength(contour,True)
    # bounding box: x,y,width,height
    f['BoundingBox'] = cv2.boundingRect(contour)
    # centroid    = m10/m00, m01/m00 (x,y)
    if (m['m00'] == 0):
        f['Centroid'] = 'undefined'
    else:
        f['Centroid'] = ( m['m10']/m['m00'],m['m01']/m['m00'] )
    
    # EquivDiameter: diameter of circle with same area as region
    f['EquivDiameter'] = np.sqrt(4*f['area']/np.pi)
    # Extent: ratio of area of region to area of bounding box
    f['Extent'] = f['area']/(f['BoundingBox'][2]*f['BoundingBox'][3])
    return f

# fs - listof functions
def comp(*fs):
    def f(f1, f2):
        return lambda x: f1(f2(x))
    return reduce(f, fs)

def nestedFor(array, f):
    aList = [list(a) for a in array]
    for i in range(len(aList)):
        for j in range(len(aList[0])):
            aList[i][j] = f(aList[i][j])
    return np.array(aList, dtype = array.dtype)

def likelyhood(l1, l2):
    def f(x,y):
        return min(x,y) / max(x,y)
    return reduce(mul, map(f, l1, list(l2)), 1)

def identity(x): return x

def const(x): return lambda *_ : x

indent = 0
def logged(f, name = "funName"):
    if (name == "funName"):
        name = f.func_name
    def loggedF(*args):
        global indent
        print indent*" ", name, "received: ", str(args)
        indent = indent + 1
        r = apply(f, args)
        indent = indent -1
        print indent*" ", "returning: " + str(r)
        return r
    return loggedF

def getCircularity(per, area):
    return (per*per) / area

def isCircle(per, area):
    if (getCircularity(per, area) < 16):
        return True
    return False

allImagesNames = ["P1000697s.jpg", "P1000698s.jpg", "P1000699s.jpg", 
                  "P1000703s.jpg", "P1000705s.jpg", "P1000706s.jpg", 
                  "P1000709s.jpg", "P1000710s.jpg", "P1000713s.jpg"]

allImages = map(lambda im:"../images/trainingSet/" + im, allImagesNames)

allExceptedValues = [208, 133, 78, 67, 162, 167, 130, 130, 170]

def getKernel(n = 7):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n,n))

def getKernel(n = 7):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n,n))

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
    returns all the contours and a list of the pairs(indexes, area) that meet the 
    requirements """
def findAllContourByHolesArea(gray, minArea = 1700, maxArea = 1000000000, 
                              inclFilled = False):    
    contour,hier = cv2.findContours(np.array(gray, np.uint8), 
                                    cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
 
    idxs = []   
    for i, cnt in enumerate (contour):
        area = sumHolesArea (hier, contour, i)
        
        if (hier[0][i][2] == -1):
            if (inclFilled):
                idxs.append((i, area))
        elif (area > minArea) & (area < maxArea):
            idxs.append((i, area))

    return (contour, idxs)


""" ----------------  utils: printing """

def imwrite(filename):
    def f(im):
        if (max(map(max,im)) <= 1):
            im = im * 255
        print ("writing to " + filename + " ...")
        if (not cv2.imwrite(filename, im)):
            print ("image " + filename + " was not written")
        else:
            print (filename + " written")
        return im
    return f

def showImg(im):
    winName = '__'
    cv2.namedWindow(winName)
    cv2.imshow(winName, im)
    cv2.waitKey(0)
    cv2.destroyWindow(winName)
    return im
    
""" a wrapper around a combinator 'f' and array of images. Then read, 
    apply function 'f' to the read image and print the result to a named window """
def showImgs(f, imgs):    
    map(comp(showImg, f, cv2.imread), imgs)

def printArr(img):
    print "printing arr: "
    print img
    return img

def printMaxMinVals(im):
    maxV = max(map(max,im))
    minV = min(map(min,im))
    print "(max = %s, min = %s) " % (maxV, minV)
    return im

def printMaxMinIdxs(im):
    maxIdx = (0,0)
    minIdx = (0,0)
    for i, v in enumerate(im):
        for j, v1 in enumerate(im[i]):
            if (im[i][j] > im[maxIdx[0]][maxIdx[1]]):
                maxIdx = (i,j)
            if (im[i][j] < im[minIdx[0]][minIdx[1]]):
                minIdx = (i,j)            
    maxV = im[maxIdx[0]][maxIdx[1]]
    minV = im[minIdx[0]][minIdx[1]]
    print "(max = %s, min = %s) " % (maxV, minV)
    print "(maxIdx = %s, minIdx = %s) " % (maxIdx, minIdx)
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

def printText(imToText):
    def f(im):
        print (imToText(im))
        return im
    return f
    
""" ----------------  utils: image processing basic stuff """

#another way: im_gray = cv2.imread('grayscale_image.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
def mycvtConvert(color = cv2.COLOR_BGR2GRAY, t = np.uint8):
    return lambda im : cv2.cvtColor(np.array(im, t), color)

def normalizeGS(img):
    return np.array(img, np.uint8)

def mkThresholdFn(tr = 0):
    def tresholdFn(ims):
        if (tr == 0):
            return cv2.threshold(ims, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #(treshold, _) = cv2.threshold(ims, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return cv2.threshold(ims, tr, 255, cv2.THRESH_BINARY)[1]
    return tresholdFn

def myContours1(im):
    im = np.array(im)
    (contours , hierarchy1)= cv2.findContours(np.array(im), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow(winName, contours[x])
    # showImg(contours)
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
def openMO(kernel = getKernel(), iterations = 1):
    return lambda im: cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel, 
                                       iterations = iterations)

def myContours(im):
    im = np.array(im)
    (contours , hierarchy1)= cv2.findContours(im, 1 , 1)
    cv2.drawContours(im, contours, -1, 150,  hierarchy = hierarchy1)
    return im
    
# n must be 0, 1 or 2
def splitFn(n): 
    def f(im):
        return cv2.split(im)[n]
    return f

def intensityLikelyhood(img, intensity):
    img = np.cast[np.int32](img)
    
    img = abs(img - intensity)

    maxV = float(max(map(max, img)))
    minV = float(min(map(min, img)))
    diff = maxV - minV

    img = 1.0 - ((img - minV) / diff)
    
    return img

def splitterByColor(aColor):
    def f(im): # im :: [[[Int]]]
        # after applying this function each element will be scaled from 0 to 1
        def g(img, color):
            img = intensityLikelyhood(img, color)
            return img*img*img*img
        def h(*lst):
            #return 255*reduce(mul, lst, 1)
            return 255 * (reduce(add, lst, 0) / len(lst))
        (im3, im2, im1) = cv2.split(im)
        (c1,  c2,  c3)  = aColor
        (im1, im2, im3) = (g(im1, c1), g(im2, c2), g(im3, c3))
        resImg = normalizeGS(h(im1,im2,im3))
        
        return resImg
    return f
    
def copperSplitter(): 
    copperColor = (170, 70, 30)
    return splitterByColor(copperColor)

def invert(img):
    maxV = max(map(max, img))
    if (maxV == 0):
        maxV = 1
    t = img.dtype
    return np.array(maxV-img, t)

def yetAnotherCoinsSplitter(img):
    copperColor = (170, 70, 30)
    img = intensityLikelyhood(cv2.split(img)[2], copperColor[0])
    img = img*img*img*img*255
    return normalizeGS(img)

""" ----------------  masks combinators  """

def sumMasks(m1, m2):
    return m1 + m2

def combineMasks(combinator, mFn1, mFn2, type1 = int, type2 = np.float64):
    def f(im): 
        im1, im2 = np.array(mFn1(im), type1), np.array(mFn2(im), type1)
        return np.array(combinator(im1, im2), type2)
    return f

""" ----------------  utils: image processing complicated stuff """

def itemsWithBigHoles(orig):
    # 1500, 2100 min and max sizes of the holes of the red stuff
    (contour, idxs) = findAllContourByHolesArea(deepcopy(orig), 1800, 2100)
    gray = np.zeros((len (orig), len (orig[0])))
    
    cntIdx = 0
    color = 1
    thickness = -1 # Thickness of lines the contours are drawn with. If it is negative (for example, thickness=CV_FILLED ), the contour interiors are drawn.
    map(lambda (i, _) : cv2.drawContours(gray, [contour[i]], cntIdx, color, thickness),
        idxs)

    return gray

def fillSmallHoles(minA = 0, maxA = 350000, inclFilled = False):
    def f(orig):
        (contour, idxs) = findAllContourByHolesArea(deepcopy(orig), minA, maxA, inclFilled)
        gray = np.zeros((len (orig), len (orig[0])))
        
        cntIdx = 0
        color = 1
        thickness = -1 # Thickness of lines the contours are drawn with. If it is negative (for example, thickness=CV_FILLED ), the contour interiors are drawn.
        map(lambda (i, _) : cv2.drawContours(gray, [contour[i]], cntIdx, color, thickness),
            idxs)
    
        return gray
    return f

import cv2
import numpy as np
from numpy import array
from copy import copy, deepcopy

""" --------------------------  useful notes   -------------

    hier [[[next, previous, firstChild, parent]]]
    getXY = lambda p : (p[0][0], p[0][1]) # kinda weird 
    sys.path.append('G:\sting\univer\master 2\IMAGE PROCESSING\Lab 1\images')    
"""

deleteme = []

""" ----------------  utils, and local constants """
# fs - listof functions
def comp(fs):
    def f(f1, f2):
        return lambda x: f1(f2(x))
    return reduce(f, fs)

def identity(x): return x

def const(x): return lambda *_ : x

allImages = ["P1000697s.jpg", "P1000698s.jpg", "P1000697s.jpg", 
            "P1000699s.jpg", "P1000703s.jpg", "P1000705s.jpg", 
            "P1000706s.jpg", "P1000709s.jpg", "P1000710s.jpg", "P1000713s.jpg"]

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
def findAllContourByHolesArea(gray, minArea = 1700, maxArea = 1000000000):    
    contour,hier = cv2.findContours(gray,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
 
    idxs = []   
    for i, cnt in enumerate (contour):
        area = sumHolesArea (hier, contour, i)
        if (hier[0][i][2] != -1) & (area > minArea) & (area < maxArea):
            idxs.append((i, area))

    return (contour, idxs)

""" ----------------  utils: printing """

""" a wrapper around a combinator 'f' and array of images. Then read, 
    apply function 'f' to the read image and print the result to a named window """
def showImgs(f, imgs):
    def showImg(im):
        winName = '__'
        cv2.namedWindow(winName)
        cv2.imshow(winName, im)
        cv2.waitKey(0)
        cv2.destroyWindow(winName)
        
    map(comp([showImg, f, cv2.imread]), imgs)

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
    print im
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

""" ----------------  utils: image processing basic stuff """

#another way: im_gray = cv2.imread('grayscale_image.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
def mycvtConvert(color = cv2.COLOR_BGR2GRAY, t = np.uint8):
    return lambda im : cv2.cvtColor(np.array(im, t), color)

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

def invert(maxV = 1):
    v1 = maxV
    v2 = maxV+maxV
    def f(img): 
        t = img.dtype
        return np.array(((img+1)%(maxV+1))*maxV, t)
    return f

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

def fillSmallHoles(orig):
    (contour, idxs) = findAllContourByHolesArea(deepcopy(orig), 0, 350000)
    gray = np.zeros((len (orig), len (orig[0])))
    
    cntIdx = 0
    color = 1
    thickness = -1 # Thickness of lines the contours are drawn with. If it is negative (for example, thickness=CV_FILLED ), the contour interiors are drawn.
    map(lambda (i, _) : cv2.drawContours(gray, [contour[i]], cntIdx, color, thickness),
        idxs)

    return gray

""" ----------------  feature extraction """

def getFeatures(contour):
    m = cv2.moments(contour)
    f = {}
    
    f['area'] = m['m00']
    f['perimeter'] = cv2.arcLength(contour,True)
    # bounding box: x,y,width,height
    f['BoundingBox'] = cv2.boundingRect(contour)
    # centroid    = m10/m00, m01/m00 (x,y)
    f['Centroid'] = ( m['m10']/m['m00'],m['m01']/m['m00'] )
    
    # EquivDiameter: diameter of circle with same area as region
    f['EquivDiameter'] = np.sqrt(4*f['area']/np.pi)
    # Extent: ratio of area of region to area of bounding box
    f['Extent'] = f['area']/(f['BoundingBox'][2]*f['BoundingBox'][3])
    f['circularity'] = (f['perimeter']*f['perimeter']) / f['area']
    return f

""" ----------------  utils: image labeling """

""" fn - is a function that receives image and returns a list which contains
    tuples - text and point. 
    Normally one can call it with const(["text", (x, y)])"""
def labelText(labeller):
    def f(img):
        def g((text, point)):
            deleteme.append(float(text))
            print text
            cv2.putText(img, text[:5], point, cv2.FONT_ITALIC, 0.5, 
                        0, 2)
        map(g, labeller(img))
        
        return img
    return f
    
def mkAreaLabeller(minArea, maxArea):
    def f(orig):        
        contours, hier = cv2.findContours(np.array(orig, np.uint8), 
                                          cv2.RETR_CCOMP, 
                                          cv2.CHAIN_APPROX_SIMPLE)
        
        def g(contour):
            f = getFeatures(contour)
            text = str(f['area'])
            point = (int(f['Centroid'][0]), int(f['Centroid'][1]))
            area = f['area']
            if (area < maxArea) & (minArea < area):
                return [(text, point)]
            return []
        
        return [x for a1 in map(g, contours) for x in a1]
        
    return f

""" ----------------  coins counting """
coinsAreas = [(1, 10000), (2, 14000), (5, 17000), (10, 15000),
              (20, 19000), (50, 23000), (100, 21000)]

def areaToCoinValue(area):
    def f((a,b)) : return (b,a)
    a = sorted(map(f, coinsAreas))
    (coinA, coinV) = (0,1) # indexes of area and value
    
    v1 = filter(lambda (v, _) : v <= area, a)
    v2 = filter(lambda (v, _) : v > area, a)
    if v1 == []: # area is less than minn elem
        return a[0][coinV]
    if v2 == []: # area is more than max elem
        return a[-1][coinV] # returning the last elem
        
    (leftN, rightN) = (v1[-1], v2[0]) # left and right neighbours
    (dist1, dist2) = (area-leftN[coinA], rightN[coinA]-area)
    if (dist1 < dist2):# lesser
        return leftN[coinV]
    else:
        return rightN[coinV]

def coinsSum(listOfAreas):
    return sum(map(areaToCoinValue, listOfAreas))

""" ----------------  resulting filters and entry point """

def addRedStuffFilter(maskFn):
    redStuffFilter = comp([invert(), 
                           itemsWithBigHoles, closeMO(getKernel(5)), 
                           mkThresholdFn(180), splitFn(2)])
    return combineMasks(cv2.bitwise_and, maskFn, redStuffFilter)

def run():
    n = 10

    """ making a mask for coins """
    
    """ for all the coins except red(brown?) one,
        both close and open operations was not able to do what I wanted """
    defaultMaskFn = comp([#labelText(const([("asd",(110,110))])), 
                          labelText(mkAreaLabeller(2000, 200000000)), 
                          #printTypes,
                          #mycvtConvert(cv2.COLOR_GRAY2BGR),
                          #printTypes,
                          fillSmallHoles, mkThresholdFn(), 
                          mycvtConvert()])
    """ 2 is a red color. for getting red coins """
    redCoinsMask = comp([invert(255), 
                         mkThresholdFn(110), splitFn(0)])
    """ combination of both """
    maskFn = combineMasks(cv2.bitwise_or, defaultMaskFn, redCoinsMask)
    filteredMaskFn = addRedStuffFilter(maskFn)
    #maskFn = comp([erode(), erode(), erode(), mkMaskCombinator(defaultMaskFn, redStuffFilter)])

    # idk why mb will be useful later
    generalContoursFn = comp([myContours, mkThresholdFn(), splitFn(2)])

    # showImgs(lambda x:x, allImages[:])
    # showImgs(maskFn, allImages[:])

    startN, lastN = 0, 1

# 8, 5, 8, 5, 4, 4, 5, 3, 3, 3, 

    showImgs(lambda x:x, allImages[startN:lastN])
    #showImgs(redCoinsMask, allImages[startN:lastN])
    #showImgs(redStuffFilter, allImages[startN:lastN])
    showImgs(defaultMaskFn, allImages[startN:lastN])
    #showImgs(filteredMaskFn , allImages[startN:lastN])

    print(coinsSum (deleteme))
    
    
    [(1, 10017.0), (3, 13967.0), (5, 17115.0), 
     (10, 15040.5), (20, 18918.5), (20, 19390.0), (50, 22790.5), (100, 20872.0)]
    # showImgs(someFun1, allImages[8:])
    # showImgs(redStuffFilter, allImages[3:4])
     
    # showImgs(generalContoursFn, allImages[:])
    # showImgs(maskFn, allImages[3:4])
    # showImgs(redStuffFilter, allImages[3:4])
    
    # showImgs(generalFn, allImages[:1])


run()
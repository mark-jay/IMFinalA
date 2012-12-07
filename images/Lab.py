import cv2
import numpy as np
from numpy import array
from copy import copy, deepcopy
from math import pi, sqrt
from operator import itemgetter

from Utils import *
from CoinsCounting import *
 
""" --------------------------  useful notes   -------------

    hier [[[next, previous, firstChild, parent]]]
    getXY = lambda p : (p[0][0], p[0][1]) # kinda weird 
    sys.path.append('G:\sting\univer\master 2\IMAGE PROCESSING\Lab 1\images')    
"""

deleteme = []

#FIXME delete me 
___id___ = 0
def getId():
    global ___id___
    ___id___ = ___id___ + 1
    return ___id___

""" ----------------  utils: image labeling """

""" fn - is a function that receives image and returns a list which contains
    tuples - text and point. 
    Normally one can call it with const(["text", (x, y)])"""
def labelText(labeller):
    def f(img):
        def g((text, point)):
            cv2.putText(img, text[:5], point, cv2.FONT_ITALIC, 0.5, 
                        0, 2)
        map(g, labeller(img))
        
        return img
    return f
    
def imgToFueaturesList(img):
    contours, hier = cv2.findContours(np.array(img, np.uint8), 
                                      cv2.RETR_CCOMP, 
                                      cv2.CHAIN_APPROX_SIMPLE)
    return map(getFeatures, contours)

def mkCoinsAreaLabeller():
    minArea, maxArea = 2000, 200000000
    def f(orig):
        def g(f):
            id_ = getId()
            print id_
            v = logged(featToValue)(f)
            if (v!=0):
                point = (int(f['Centroid'][0]), int(f['Centroid'][1]))
                text = str(id_)
                deleteme.append((f['area'], v))
                return [(text, point)]
            return []
        
        return [x for a1 in map(g, imgToFueaturesList(orig)) for x in a1]
        
    return f

""" ----------------  tests """

def testImages(f, images, expectedValues):
    def testTuple((a, e, i)):
        if a != e:
            print str(i)+")","actual = " + str(a), "expected = " + str(e)
            return False
        return True
    g = comp(featuresToCoinsSum, imgToFueaturesList, f, cv2.imread)
    actualValues = map(g, images)
    l = filter(testTuple, zip(actualValues, expectedValues, range(len(actualValues))))
    print "test passed:" + str(len(l)) + "/" + str(len(actualValues))

""" ----------------  resulting filters and entry point """

def addRedStuffFilter(maskFn):
    redStuffFilter = comp(invert(), 
                          itemsWithBigHoles, closeMO(getKernel(5)), 
                          mkThresholdFn(180), splitFn(2))
    return combineMasks(cv2.bitwise_and, maskFn, redStuffFilter)

def run():
    n = 10

    """ making a mask for coins """
    
    """ for all the coins except red(brown?) one,
        both close and open operations was not able to do what I wanted """
    defaultMaskFn = comp(fillSmallHoles, mkThresholdFn(), 
                         mycvtConvert())
    """ 2 is a red color. for getting red coins """
    redCoinsMask = comp(invert(255), 
                         mkThresholdFn(110), splitFn(0))
    """ combination of both """
    maskFn = comp(fillSmallHoles, 
                  combineMasks(cv2.bitwise_or, defaultMaskFn, redCoinsMask))
    filteredMaskFn = addRedStuffFilter(maskFn)
    #maskFn = comp([erode(), erode(), erode(), mkMaskCombinator(defaultMaskFn, redStuffFilter)])

    # idk why mb will be useful later
    generalContoursFn = comp(myContours, mkThresholdFn(), splitFn(2))

    # showImgs(lambda x:x, allImages[:])
    # showImgs(maskFn, allImages[:])

    startN = 0
    lastN = 99

# 8, 5, 8, 5, 4, 4, 5, 3, 3, 3, 

    #  showImgs(lambda x:x, allImages[startN:lastN])
    #showImgs(redCoinsMask, allImages[startN:lastN])
    #showImgs(redStuffFilter, allImages[startN:lastN])
    
    #showImgs(comp(labelText(mkCoinsAreaLabeller()), maskFn), 
    #              allImages[startN:lastN])
    #showImgs(filteredMaskFn , allImages[startN:lastN])

    #def f(img) : print img
    #map(f, sorted(allCoinsCombos, key=itemgetter(2)))
    
    """
    for i in range (len (allImages)):#len (allImages)):
        global deleteme
        deleteme = []
        showImgs(lambda x:x, allImages[i:i+1])
        showImgs(comp(labelText(mkCoinsAreaLabeller()), maskFn), 
                      allImages[i:i+1])
    """ 
    
    showImgs(lambda x:x, allImages[0:4])
    showImgs(comp(labelText(mkCoinsAreaLabeller()), maskFn), 
                  allImages[0:4])
    print(deleteme[::-1])
    testImages(maskFn, allImages, allExceptedValues)
"""
[(17687.5, 5), (10347.0, 1), (23194.5, 50), (14316.5, 2), (15191.0, 10), 
 (19624.0, 20), (19271.5, 20), (21027.5, 100)]
[(14110.5, 2), (15240.0, 10), (10284.5, 1), (19248.5, 20), (20928.0, 100)]
[(17687.5, 5), (10347.0, 1), (23194.5, 50), (14316.5, 2), (15191.0, 10), 
 (19624.0, 20), (19271.5, 20), (21027.5, 100)]
[(23082.0, 50), (13955.0, 2), (10012.5, 1), (17212.0, 5), (19189.0, 20)]
[(23690.0, 50), (14373.5, 2), (15794.5, 10), (18106.0, 20), (29757.5, 50)]
[(13806.5, 2), (43957.5, 50), (15395.5, 10)]
[(22997.5, 50), (31096.5, 50), (15377.0, 10), (21075.0, 100)]
[(4505.0, 1), (2336.0, 1), (17314.5, 5), (15258.5, 10), (21074.5, 100), 
 (24347.5, 50), (19141.5, 20)]
[(20415.5, 100), (15864.0, 10), (21859.0, 100), (57308.0, 50), (20058.0, 100)]
[(22257.0, 50), (21285.5, 100), (42094.5, 50)]
[(22257.0, 50), (21285.5, 100), (42094.5, 50)]
"""
"""
    for i, image in enumerate(allImages):
        global deleteme
        deleteme = []
        showImgs(comp(labelText(mkCoinsAreaLabeller()), maskFn), 
                      [image])
        print(deleteme[::-1])
""" 
    
   
    # showImgs(someFun1, allImages[8:])
    # showImgs(redStuffFilter, allImages[3:4])
     
    # showImgs(generalContoursFn, allImages[:])
    # showImgs(maskFn, allImages[3:4])
    # showImgs(redStuffFilter, allImages[3:4])
    
    # showImgs(generalFn, allImages[:1])


run()
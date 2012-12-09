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
    redStuffFilter = comp(invert, 
                          itemsWithBigHoles, closeMO(getKernel(5)), 
                          mkThresholdFn(180), splitFn(2))
    return combineMasks(cv2.bitwise_and, maskFn, redStuffFilter)

def run():
    n = 3

    """ making a mask for coins """
    
    """ for all the coins except red(brown?) one,
        both close and open operations was not able to do what I wanted """
    defaultMaskFn = comp(fillSmallHoles(), mkThresholdFn(), 
                         mycvtConvert())
    """ 2 is a red color. for getting red coins """
    f = dilate
    op = cv2.MORPH_ELLIPSE
    redCoinsMask = comp(#f(cv2.getStructuringElement(op, (n,n))), 
                        #invert, 
                        #fillSmallHoles(0, 1000000), 
                        #mkThresholdFn(190), #splitFn(2))
                        
                        #mkThresholdFn(), #showImg,
                        invert, showImg,
                        copperSplitter()
                        )
    """ combination of both """
    maskFn = comp(#closeMO(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n,n))),
                  combineMasks(cv2.bitwise_or, defaultMaskFn, redCoinsMask))

    filteredMaskFn = addRedStuffFilter(maskFn)

    generalContoursFn = comp(myContours, mkThresholdFn(), splitFn(2))

    startN = 0
    lastN = 1
    
    #showImgs(identity, allImages[startN:lastN])
    #showImgs(defaultMaskFn, allImages[startN:lastN])
    showImgs(redCoinsMask, allImages[startN:lastN])
    #showImgs(maskFn, allImages[startN:lastN])
    print(deleteme[::-1])

run()
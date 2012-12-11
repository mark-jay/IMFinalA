import cv2
import numpy as np
from numpy import array
from copy import copy, deepcopy
from math import pi, sqrt
from operator import itemgetter

from Utils import allExceptedValues, allImages # for editor
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

def onlyIdxs(idxs, lst):
    filtered = filter(lambda (i,v): i in idxs, zip(range(len(lst)), lst))
    return map(itemgetter(1), filtered)

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
        def g(feat):
            v = featToValue(feat)
            if (feat['area']>100):
                point = (int(feat['Centroid'][0]), int(feat['Centroid'][1]))
                text = str(v)
                deleteme.append((feat, text))
                print (text, point)
                return [(text, point)]
            return []
        
        return [x for a1 in map(g, imgToFueaturesList(orig)) for x in a1]
    return f

""" ----------------  tests """

def testImages(f, images, expectedValues):
    print "\n" + "starting tests"
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

def mixedShow(images, functions):
    [showImgs(f, [i]) for i in images for f in functions]

def run():
    n = 3

    """ making a mask for coins """
    
    """ for all the coins except red(brown?) one,
        both close and open operations was not able to do what I wanted """
    defaultMaskFn = comp(fillSmallHoles(), mkThresholdFn(), 
                         mycvtConvert())
    """ 2 is a red color. for getting red coins """
    redCoinsMask = comp(mkThresholdFn(), yetAnotherCoinsSplitter)
    """ combination of both """
    maskFn = comp(#closeMO(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n,n))),
                  fillSmallHoles(0, 1000000, inclFilled = True),
                  combineMasks(cv2.bitwise_or, defaultMaskFn, redCoinsMask))

    generalContoursFn = comp(myContours, mkThresholdFn(), splitFn(2))

    redStuffFilter = comp(invert, 
                          itemsWithBigHoles,
                          #showImg, closeMO(getKernel(5)), 
                          closeMO(getKernel(5)), 
                          mkThresholdFn(180), splitFn(2))

    filteredMaskFn = combineMasks(cv2.bitwise_and, maskFn, redStuffFilter)
    
    joined = onlyIdxs([4,5,8], allImages)
    icImages = onlyIdxs([2,3], allImages) # invisble coins images
    rsImages = onlyIdxs([7,8], allImages) # red stuff images
    failed =   onlyIdxs([3,6,7,8], allImages)
    
    mixedShow(allImages[8:9],
              [#identity,
               #defaultMaskFn,
               #redCoinsMask,
               #maskFn,
               #redStuffFilter,
               comp(labelText(mkCoinsAreaLabeller()), filteredMaskFn),
               #temp,
               identity,
               ])
    
    testImages(filteredMaskFn, allImages[8:9], allExceptedValues[8:9])
    print(deleteme[::-1])

run()
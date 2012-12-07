import cv2
import numpy as np
from numpy import array
from copy import copy, deepcopy
from math import pi, sqrt
from operator import itemgetter

from Utils import *

""" ----------------  coins counting """
[(1, 10017.0), (2, 13967.0), (5, 17115.0), 
 (10, 15040.5), (20, 18918.5), (20, 19390.0), (50, 22790.5), (100, 20872.0)]

coinsAreas = [(1, 10017.0), (2, 13967.0), (5, 17650.0), (10, 15000),
              (20, 19500), (50, 23000), (100, 21000)]

def allCoinsCombosMaker(coinsAreas):

    def areaToRad(area):
        # S = Pi*r*r,
        r = sqrt(area/pi)
        return r
    
    def radToPer(r):
        return 2*pi*r

    # circularity of sum:
    # (P1+P2)^2 / (S1+S1) = 
    # (((2*pi*r1) + (2*pi*r2))^2) / ((pi*r1*r1) + (pi*r2*r2)) =
    # ((2*pi*(r1 + r2)))^2 / pi(r1*r1 + r2*r2) =
    # 4*pi*pi*(r1 + r2)^2 / pi(r1*r1 + r2*r2) =
    # 4*pi*(r1 + r2)^2 / (r1*r1 + r2*r2)
    def circOfSum(r1, r2):
        return (4*pi*(r1 + r2)*(r1 + r2)) / (r1*r1 + r2*r2)
    
    def expandTuple(c):
        v, area = c
        per = radToPer(areaToRad(area))
        return (v, area, per, getCircularity(per, area))

    # 2 pair each of which contains value of the coin and its area
    def mkCombo((c1, c2)):
        (v1, area1) = c1
        (v2, area2) = c2
        (r1, r2) = (areaToRad(area1), areaToRad(area2))
        (per1, per2) = (radToPer(r1), radToPer(r2))
        return (v1+v2, area1+area2, per1+per2, circOfSum(r1,r2))
    
    def mkAllPair(lst):
        n = len( lst )
        acc = []
        for i in range(n):
            for j in range(i,n):
                acc.append((lst[i], lst[j]))
        return acc

    return map(mkCombo, mkAllPair(coinsAreas)) + map(expandTuple, coinsAreas)

allCoinsCombos = allCoinsCombosMaker(coinsAreas)

def shapeToValue((shapeArea, shapePerimeter)):
    if (isCircle(shapePerimeter, shapeArea)):
        return areaToCoinValue(shapeArea)
    return 0 # FIXME: must be msth else

""" [(Float, Float)], an area and a perimeter """
def coinsSum(listOfAreasPerimeters):
    return sum(map(areaToCoinValue, listOfAreasPerimeters))

def featToValue(f):
    fPer = f['perimeter']
    fArea = f['area']
    if (fArea*fPer > 0):                     # an object
        fCirc= getCircularity(fPer, fArea)
        if ((fCirc < 18.5) |                  # a single coin
            ((fCirc > 20.) & (fCirc < 26.))):  # two coins
            def f((v, area, per, circ)):
                areaC = min(fArea, area) / max(fArea, area) # coefficient
                perC = min(fPer, per) / max(fPer, per)      # coefficient
                circC = min(fCirc, circ) / max(fCirc, circ)     # coefficient
                coef = areaC * perC * circC
                return (coef, v)
            (coef,v) = sorted(map(f, allCoinsCombos), reverse=True)[0]
            if (coef > 0.1):
                return v
    return 0

def featuresToCoinsSum(fs):
    return sum(map(featToValue, fs))

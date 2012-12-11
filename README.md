IMFinalA
========

Final project A for the Image processing subject

Description of the image processing
===================================

maskFn
------

In order to get an binary image which represent all coins we used maskFn:

    maskFn = comp(fillSmallHoles(0, 1000000, inclFilled = True),
                  combineMasks(cv2.bitwise_or, defaultMaskFn, redCoinsMask))

Where comp function is just a composition of functions given as parameters. Here we combined defaultMaskFn and redCoinsMask using bitwise_or operation and redirects
the result to function fillSmallHoles. 
FillSmallHoles function as its name suggests for all the found objects in the image calculates a sum of all holes. If the sum is between 0 and 1000000(in our case)
then FillSmallHoles function fills all the holes in the image

defaultMaskFn
-------------

DefaultMaskFn is a mask, that tries to find all the coins except for red ones:

    defaultMaskFn = comp(fillSmallHoles(), mkThresholdFn(), 
                         mycvtConvert())

Where:
 -mycvtConvert is a cv2.cvtColor function that uses cv2.COLOR_BGR2GRAY as a color.
 -mkThresholdFn is a threshold, using otsu method
 -fillSmallHoles is the same function we considered before

redCoinsMask
------------

RedCoinsMask is a mask, that tries to find all copper coins:

    redCoinsMask = comp(mkThresholdFn(), yetAnotherCoinsSplitter)

Where yetAnotherCoinsSplitter is a function that converts a colored image to a grayscale one. That function considers only red component of the color. 
It tries to match that intesity of the red component to the fixed value 170 which is red component of the copper color, using intensityLikelyhood.
The returned result is a grayscale image where the more value of each pixes the more red component of the color of the pixel was like 170. I.e.
if we had image [[[r = 170, ...(g and b do not matter)], [r = 85...], [r = 0 ...]]] the result will be [[255], [128], [0]]. All the values will be scaled 
from 0 to 255

redStuffFilter
--------------

After that we introduce a redStuffFilter function:

    redStuffFilter = comp(invert, itemsWithBigHoles,
                          closeMO(getKernel(5)), mkThresholdFn(180), splitFn(2))

The purpose of this function is to filter red object with a hole, because maskFn fills all the holes, so the object with a hole which is not a coin can be
considred as a coin. So first that function select red component of the image because that kind of objects is red. Then it does a threshold using fixed value of
180, which is very high. Then it uses close morphological operation using sphere kernel with diameter = 5 pixel and then inverts the image to use bitwise_and later
The result will be a white image([[1,1...][1,1...]...]]) if no objects like that were found and white image with black round mask otherwise

filteredMaskFn
--------------

So finally we have:

    filteredMaskFn = combineMasks(cv2.bitwise_and, maskFn, redStuffFilter)

Where we can see coins and other object. We hope we can filter other objects checking features of the objects such as perimeter, area, circularity and, maybe, others


Description of the classifying algorithm
========================================

Features to classify by
-----------------------

Firstly lets define features we will use. They defined in the Utils.getFeatures(contour) most of the features(but not all of them) obtained using cv2.moments. 
They are:
 area - a float value which represents an area
 perimeter - a float value which represents a perimeter
 BoundingBox - a tuple: (x,y,width,height). Obtained by using cv2.boundingRect
 Centroid - a centroid. Could be a tuple (x,y) or string = 'undefined'. if it was impossible to get a centroid
 EquivDiameter - diameter of circle with same area as region
 Extent - ratio of area of region to area of bounding box

We will not use all the features right now, but may be later

Generating data
---------------

All the generated data can be found in images/generatedData/. We generated the images with 

Training dataSet
----------------

The module AprioryData contains all the data obtained from the training dataset. It is an array of tuples (features, tag), which are a dictionary and an integer
respectively.

Also we have tagsVals which are pairs tags and coins values. If it is 2 joined coins then a value will be the value of the first one + the value of the second one

Given that we can find mean value of each feature for all the coins and all the coins pairs

A function featsVals is the only function we'd like to export from this module and as I said this module supposed to have no logic, only data or at most average data.

Coins counting 
--------------

The module CoinsCounting supposed to export only 1 function: featToValue. This function receives features obtained by using Utils.getFeatures and 
returns the value that as close as possible to the corresponding value from the training data. Coefficient of the likelyhood of features1 and features2 
is as follow:

    areaC = min(fArea, fs['area']) / max(fArea, fs['area'])
    perC = min(fPer, per) / max(fPer, per)
    circC = min(fCirc, circ) / max(fCirc, circ)
    coef = areaC * perC * circC

So if all the features will be the same result will be 1. Otherwise it is lower and could be up to 0.
If coefficient more than 0.75 it seems like it is a coin  with value v. otherwise it's something else and value will be zero.
Circularity of single coins less than 18.5, while coupled or joined ones is between 20.0 and 28.0. Or at least we believe so.=)

So all we need to know amount of the coins so far is to apply featToValue to every object in the image after applying a filter.
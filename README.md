IMFinalA
========

Final project A for the Image processing subject

Description of the image processing
===================================

maskFn
------

In order to get an binary image which represent all coins we used maskFn:

>maskFn = comp(fillSmallHoles(0, 1000000, inclFilled = True),
>              combineMasks(cv2.bitwise_or, defaultMaskFn, redCoinsMask))

 Where comp function is just a composition of functions given as parameters. Here we combined defaultMaskFn and redCoinsMask using bitwise_or operation and redirects
the result to function fillSmallHoles. 
 FillSmallHoles function as its name suggests for all the found objects in the image calculates a sum of all holes. If the sum is between 0 and 1000000(in our case)
then FillSmallHoles function fills all the holes in the image

defaultMaskFn
-------------

DefaultMaskFn is a mask, that tries to find all the coins except for red ones:

>defaultMaskFn = comp(fillSmallHoles(), mkThresholdFn(), 
>                     mycvtConvert())

Where:
 -mycvtConvert is a cv2.cvtColor function that uses cv2.COLOR_BGR2GRAY as a color.
 -mkThresholdFn is a threshold, using otsu method
 -fillSmallHoles is the same function we considered before

redCoinsMask
------------

RedCoinsMask is a mask, that tries to find all copper coins:

>redCoinsMask = comp(mkThresholdFn(), yetAnotherCoinsSplitter)

Where yetAnotherCoinsSplitter is a function that converts a colored image to a grayscale one. That function considers only red component of the color. 
 It tries to match that intesity of the red component to the fixed value 170 which is red component of the copper color, using intensityLikelyhood.
 The returned result is a grayscale image where the more value of each pixes the more red component of the color of the pixel was like 170. I.e.
 if we had image [[[r = 170, ...(g and b do not matter)], [r = 85...], [r = 0 ...]]] the result will be [[255], [128], [0]]. All the values will be scaled 
 from 0 to 255

redStuffFilter
--------------

After that we introduce a redStuffFilter function:

>redStuffFilter = comp(invert, itemsWithBigHoles,
>                      closeMO(getKernel(5)), mkThresholdFn(180), splitFn(2))

The purpose of this function is to filter red object with a hole, because maskFn fills all the holes, so the object with a hole which is not a coin can be
considred as a coin. So first that function select red component of the image because that kind of objects is red. Then it does a threshold using fixed value of
180, which is very high. Then it uses close morphological operation using sphere kernel with diameter = 5 pixel and then inverts the image to use bitwise_and later
The result will be a white image([[1,1...][1,1...]...]]) if no objects like that were found and white image with black round mask otherwise

filteredMaskFn
--------------

So finally we have:

>filteredMaskFn = combineMasks(cv2.bitwise_and, maskFn, redStuffFilter)

Where we can see coins and other object. We hope we can filter other objects checking features of the objects such as perimeter, area, circularity and, maybe, others


Description of the image processing
===================================

in progress
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 00:47:22 2012
@author: fallen
"""

""" ----------------  coins counting """

"""
# an algorithm that producesall the data we need
for im in allImagesNames:
    f = comp(imwrite("../images/tagged/tagged_" + im), 
             comp(labelText(mkCoinsAreaLabeller()), filteredMaskFn),
             cv2.imread)
    f("../images/trainingSet/" + im)
"""

# resultsByTag :: [(feat :: Map String Float, tag :: Int)]
resultsByTag = [({'perimeter': 557.7716406583786, 'area': 21922.5, 'Centroid': (722.3105637282852, 239.0468392443076), 'Extent': 0.7676751759638617, 'BoundingBox': (637, 156, 171, 167), 'EquivDiameter': 167.07062554338091}, 47), ({'perimeter': 569.4284938573837, 'area': 23038.5, 'Centroid': (720.6995391771744, 240.03532492711474), 'Extent': 0.7745595750403442, 'BoundingBox': (633, 156, 176, 169), 'EquivDiameter': 171.27033967205486}, 46), ({'perimeter': 558.943213224411, 'area': 21571.5, 'Centroid': (368.46524349257123, 266.61334785867155), 'Extent': 0.778136498088161, 'BoundingBox': (286, 184, 166, 167), 'EquivDiameter': 165.72774915280351}, 45), ({'perimeter': 1051.1168735027313, 'area': 42522.0, 'Centroid': (581.9370247558127, 489.11762930561434), 'Extent': 0.5407378206187927, 'BoundingBox': (435, 359, 299, 263), 'EquivDiameter': 232.68152466671819}, 44), ({'perimeter': 716.8326100111008, 'area': 18800.0, 'Centroid': (650.9198936170213, 166.09303191489363), 'Extent': 0.4630998127894374, 'BoundingBox': (549, 52, 199, 204), 'EquivDiameter': 154.71555655790098}, 43), ({'perimeter': 465.4457380771637, 'area': 15193.0, 'Centroid': (448.623390157748, 241.19553961254084), 'Extent': 0.7751530612244898, 'BoundingBox': (379, 172, 140, 140), 'EquivDiameter': 139.08388980453964}, 42), ({'perimeter': 549.0437175035477, 'area': 21030.0, 'Centroid': (774.3811539071168, 352.72118402282456), 'Extent': 0.7819006543723974, 'BoundingBox': (693, 271, 164, 164), 'EquivDiameter': 163.63443288556499}, 41), ({'perimeter': 723.5533834695816, 'area': 28602.0, 'Centroid': (443.8197270587138, 447.21479849893944), 'Extent': 0.6055255636710066, 'BoundingBox': (327, 347, 235, 201), 'EquivDiameter': 190.83290454875734}, 40), ({'perimeter': 519.3868629932404, 'area': 18937.0, 'Centroid': (678.3462181619756, 531.0316047948461), 'Extent': 0.7731912461211824, 'BoundingBox': (601, 453, 156, 157), 'EquivDiameter': 155.27825752065152}, 39), ({'perimeter': 1119.6437846422195, 'area': 987.5, 'Centroid': (471.9777215189873, 603.6025316455696), 'Extent': 0.03498919321121072, 'BoundingBox': (388, 560, 167, 169), 'EquivDiameter': 35.458765494951642}, 38), ({'perimeter': 275.63960909843445, 'area': 4327.5, 'Centroid': (986.7351819757366, 29.39634122857693), 'Extent': 0.7899780941949617, 'BoundingBox': (940, 1, 83, 66), 'EquivDiameter': 74.228997904063192}, 37), ({'perimeter': 1000.5971019268036, 'area': 760.0, 'Centroid': (622.8796052631579, 202.54342105263157), 'Extent': 0.02246858832224686, 'BoundingBox': (549, 55, 165, 205), 'EquivDiameter': 31.107266900175009}, 36), ({'perimeter': 415.2274839878082, 'area': 299.5, 'Centroid': (702.7634947134112, 141.17084028937117), 'Extent': 0.02181672494172494, 'BoundingBox': (652, 55, 96, 143), 'EquivDiameter': 19.527806933913016}, 35), ({'perimeter': 463.10259318351746, 'area': 15230.0, 'Centroid': (458.0851389800831, 287.90182753337706), 'Extent': 0.7715298885511651, 'BoundingBox': (388, 218, 140, 141), 'EquivDiameter': 139.25314454731904}, 34), ({'perimeter': 546.4579312801361, 'area': 21222.5, 'Centroid': (779.015078336671, 357.92860564652295), 'Extent': 0.7795224977043159, 'BoundingBox': (697, 276, 165, 165), 'EquivDiameter': 164.38164811846241}, 33), ({'perimeter': 775.9970360994339, 'area': 24432.5, 'Centroid': (369.23297520379276, 473.76248166717824), 'Extent': 0.5385403808852054, 'BoundingBox': (263, 368, 214, 212), 'EquivDiameter': 176.37580666503516}, 32), ({'perimeter': 521.3868651390076, 'area': 19253.0, 'Centroid': (674.4116934157447, 568.310886615073), 'Extent': 0.7761428686608078, 'BoundingBox': (597, 490, 157, 158), 'EquivDiameter': 156.56845453278922}, 31), ({'perimeter': 573.4284937381744, 'area': 23183.5, 'Centroid': (700.2931251392872, 137.97790813869057), 'Extent': 0.7746165926024926, 'BoundingBox': (614, 52, 173, 173), 'EquivDiameter': 171.80846598863411}, 30), ({'perimeter': 924.2051854133606, 'area': 31302.0, 'Centroid': (506.96491704470424, 296.6735032905246), 'Extent': 0.5430603747397641, 'BoundingBox': (397, 163, 220, 262), 'EquivDiameter': 199.63703120738913}, 29), ({'perimeter': 469.5878745317459, 'area': 15421.0, 'Centroid': (256.26584959903164, 356.53626007824823), 'Extent': 0.7647788137274351, 'BoundingBox': (186, 286, 142, 142), 'EquivDiameter': 140.12361335392742}, 28), ({'perimeter': 546.700572013855, 'area': 21270.0, 'Centroid': (783.562787964269, 443.9117144648174), 'Extent': 0.7765607886089814, 'BoundingBox': (701, 362, 166, 165), 'EquivDiameter': 164.56550402960187}, 27), ({'perimeter': 445.5462440252304, 'area': 13990.5, 'Centroid': (464.90858082270114, 202.80771475882443), 'Extent': 0.7733830845771145, 'BoundingBox': (398, 136, 134, 135), 'EquivDiameter': 133.46631728873504}, 26), ({'perimeter': 1095.886425614357, 'area': 44322.0, 'Centroid': (677.6022742656017, 384.70760570371374), 'Extent': 0.5987436676798379, 'BoundingBox': (515, 272, 329, 225), 'EquivDiameter': 237.55530535383099}, 25), ({'perimeter': 472.4163017272949, 'area': 15490.0, 'Centroid': (254.85278674413598, 360.6574026253497), 'Extent': 0.7628287205751995, 'BoundingBox': (184, 290, 143, 142), 'EquivDiameter': 140.43674927862602}, 24), ({'perimeter': 568.4995603561401, 'area': 22961.0, 'Centroid': (644.9606434098398, 151.81849658115937), 'Extent': 0.7716426939104718, 'BoundingBox': (559, 66, 172, 173), 'EquivDiameter': 170.98202591694857}, 23), ({'perimeter': 439.54624462127686, 'area': 13881.5, 'Centroid': (325.11492514017453, 178.02048289690114), 'Extent': 0.7847532364746452, 'BoundingBox': (259, 112, 133, 133), 'EquivDiameter': 132.94538254577014}, 22), ({'perimeter': 463.102591753006, 'area': 15065.0, 'Centroid': (476.10806505144376, 279.5916583692886), 'Extent': 0.7686224489795919, 'BoundingBox': (407, 210, 140, 140), 'EquivDiameter': 138.49676437171817}, 21), ({'perimeter': 493.83051466941833, 'area': 17216.5, 'Centroid': (730.1589269983252, 364.0024879234068), 'Extent': 0.7703131991051454, 'BoundingBox': (656, 290, 150, 149), 'EquivDiameter': 148.05650482816662}, 20), ({'perimeter': 722.2396750450134, 'area': 28611.0, 'Centroid': (468.6882143231624, 468.08832267309776), 'Extent': 0.6180149044173237, 'BoundingBox': (352, 370, 235, 197), 'EquivDiameter': 190.86292624398732}, 19), ({'perimeter': 570.842707157135, 'area': 23065.0, 'Centroid': (520.1688127754895, 186.504032083243), 'Extent': 0.7751898904348995, 'BoundingBox': (435, 100, 171, 174), 'EquivDiameter': 171.36881308836951}, 18), ({'perimeter': 442.6173119544983, 'area': 13948.0, 'Centroid': (309.0852810438773, 303.4419630054488), 'Extent': 0.7710337202874517, 'BoundingBox': (242, 237, 135, 134), 'EquivDiameter': 133.2634427364311}, 17), ({'perimeter': 378.09040081501007, 'area': 10165.5, 'Centroid': (824.7874018329971, 349.7686947682521), 'Extent': 0.7754004576659039, 'BoundingBox': (768, 293, 114, 115), 'EquivDiameter': 113.76781878899365}, 16), ({'perimeter': 492.6589425802231, 'area': 17325.5, 'Centroid': (647.2677171414003, 372.9671582349716), 'Extent': 0.7751901565995526, 'BoundingBox': (573, 298, 149, 150), 'EquivDiameter': 148.52444826461758}, 15), ({'perimeter': 522.8010765314102, 'area': 19226.5, 'Centroid': (474.28184190223556, 494.26938513683365), 'Extent': 0.7701690434225285, 'BoundingBox': (396, 416, 158, 158), 'EquivDiameter': 156.46066632496041}, 14), ({'perimeter': 447.78888463974, 'area': 14088.0, 'Centroid': (287.9790365322733, 225.23327181525647), 'Extent': 0.7730041152263375, 'BoundingBox': (221, 158, 135, 135), 'EquivDiameter': 133.93057420256577}, 13), ({'perimeter': 464.6173119544983, 'area': 15236.0, 'Centroid': (618.2188347772818, 238.4297387765818), 'Extent': 0.7773469387755102, 'BoundingBox': (549, 169, 140, 140), 'EquivDiameter': 139.28057188131064}, 12), ({'perimeter': 380.6761872768402, 'area': 10257.0, 'Centroid': (804.4868057586688, 263.3941698352345), 'Extent': 0.7688905547226387, 'BoundingBox': (747, 206, 115, 116), 'EquivDiameter': 114.27868572200401}, 11), ({'perimeter': 524.8010773658752, 'area': 19344.5, 'Centroid': (455.91713406911526, 413.82337787657127), 'Extent': 0.770022291218852, 'BoundingBox': (377, 335, 159, 158), 'EquivDiameter': 156.94005980988206}, 10), ({'perimeter': 545.286359667778, 'area': 21163.5, 'Centroid': (672.3130468337782, 528.0075680613635), 'Extent': 0.7773553719008265, 'BoundingBox': (590, 446, 165, 165), 'EquivDiameter': 164.1529929821647}, 9), ({'perimeter': 496.9015825986862, 'area': 17482.0, 'Centroid': (583.5833047324867, 163.85083705144336), 'Extent': 0.771832229580574, 'BoundingBox': (509, 89, 150, 151), 'EquivDiameter': 149.19374558291682}, 8), ({'perimeter': 383.26197278499603, 'area': 10294.5, 'Centroid': (404.96935256690466, 205.93260155099), 'Extent': 0.7852402745995424, 'BoundingBox': (348, 149, 114, 115), 'EquivDiameter': 114.48739884055421}, 7), ({'perimeter': 569.7716389894485, 'area': 22954.5, 'Centroid': (784.2332004617831, 251.712431113725), 'Extent': 0.7804467564259486, 'BoundingBox': (699, 166, 171, 172), 'EquivDiameter': 170.95782266285241}, 6), ({'perimeter': 446.61731231212616, 'area': 14086.0, 'Centroid': (119.72814614984144, 273.6591059681007), 'Extent': 0.7729367866549605, 'BoundingBox': (52, 207, 136, 134), 'EquivDiameter': 133.92106715203363}, 5), ({'perimeter': 461.6883796453476, 'area': 15089.5, 'Centroid': (481.80495267128356, 361.99142891856366), 'Extent': 0.7754110996916752, 'BoundingBox': (412, 293, 140, 139), 'EquivDiameter': 138.60933630272254}, 4), ({'perimeter': 528.4579317569733, 'area': 19506.5, 'Centroid': (273.558916258683, 411.5474499953007), 'Extent': 0.7764708223867527, 'BoundingBox': (195, 333, 159, 158), 'EquivDiameter': 157.59583490491255}, 3), ({'perimeter': 518.8010765314102, 'area': 19091.5, 'Centroid': (887.7260386384864, 457.2373743987289), 'Extent': 0.7745344638727737, 'BoundingBox': (810, 379, 157, 157), 'EquivDiameter': 155.91039980806718}, 2), ({'perimeter': 546.4579312801361, 'area': 21024.5, 'Centroid': (658.2922859838125, 507.0172021530436), 'Extent': 0.7722497704315886, 'BoundingBox': (576, 425, 165, 165), 'EquivDiameter': 163.61303373596013}, 1)]

# tagsVals :: [(tag :: Int, value :: Int)]
tagsVals = dict([(1, 100),(2, 20),(3, 20),(4, 10),(5, 2),(6, 50),(7, 1),(8, 5),(9, 100),(10, 20),(11, 1),(12, 10),(13, 2),(14, 20),(15, 5),(16, 1),(17, 2),(18, 50),(20, 5),(21, 10),(22, 2),(23, 50),(24, 10),(25, 150),(26, 2),(27, 100),(28, 10),(29, 7),(30, 50),(31, 20),(33, 100),(34, 10),(39, 20),(41, 100),(42, 10),(44, 70),(45, 100)])

# featsVals :: [(feats :: Map String Float, value :: Int)]
featsVals = map(lambda (feats, coinId): (feats, tagsVals.get(coinId, 0)), 
                resultsByTag)
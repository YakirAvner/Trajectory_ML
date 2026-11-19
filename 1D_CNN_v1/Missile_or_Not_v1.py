import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.cluster import HDBSCAN


NUM_EPOCHS = 600
COPIES_PER_STEP = 6      # random pose copies of every trajectory per training step
LATENT_DIM = 32
RESAMPLE_POINTS = 32     # every trajectory -> this many evenly spaced points
ST_PASSES = 4            # iterative spatial transformer refinement passes
CONSISTENCY_WEIGHT = 1.0
CANVAS_SPREAD = 0.1      # target mean squared spread of the canonical drawing
SNAPSHOT_EVERY = 25      # frames of the training-simulation GIF
OUT_DIR = "st_output"
torch.manual_seed(0)
np.random.seed(0)
os.makedirs(OUT_DIR, exist_ok=True)

raw_trajectory_input = [
    [
        (218.33333333333334, 587.6666666666666), (218.4, 587.2), (217.8, 586.3), (217.30769230769232, 585.4615384615385), 
        (215.75, 581.75), (215.4, 581.0), (213.88888888888889, 576.2222222222222), (212.91666666666666, 574.5833333333334), (212.25, 572.25), 
        (210.66666666666666, 568.2), (209.64705882352942, 566.0), (208.1, 562.1), (206.5, 558.0), (205.21428571428572, 555.1428571428571), 
        (203.63636363636363, 550.5454545454545), (201.93333333333334, 546.9333333333333), (200.8235294117647, 543.8823529411765), 
        (198.47058823529412, 538.8823529411765), (196.25, 533.8333333333334), (194.6, 529.0), (192.8181818181818, 524.7272727272727),
        (190.8421052631579, 519.4736842105264), (189.5, 516.9090909090909), (186.5, 510.0), (186.0, 508.74074074074076), (184.5, 505.2894736842105),
        (183.38028169014083, 503.0422535211268), (181.3, 498.16), (175.0, 481.5), (173.0, 477.0), (170.33333333333334, 470.0), (167.75, 463.5), 
        (165.5, 457.5), (163.5, 451.5), (160.5, 445.5), (157.75, 438.5), (155.6, 433.4), (152.66666666666666, 426.0), (151.0, 420.0), 
        (148.0, 413.0), (145.0, 405.5), (143.0, 400.0), (140.0, 392.0), (137.0, 384.0), (135.0, 378.0), (132.0, 370.5), (129.0, 362.5), 
        (126.0, 354.5), (120.0, 338.5), (118.0, 332.5), (115.0, 324.0), (112.0, 316.0), (109.0, 308.0), (106.5, 300.5), (103.5, 293.0), 
        (100.0, 284.0), (97.33333333333333, 276.1666666666667), (94.4, 267.6), (91.5, 260.0), (88.57142857142857, 251.85714285714286), 
        (85.6, 243.4), (83.5, 237.0), (80.4, 228.6), (77.5, 220.0), (75.0, 212.0), (72.5, 205.0), (69.0, 196.0), (66.0, 187.5), 
        (64.66666666666667, 184.33333333333334), (63.333333333333336, 180.88888888888889), (61.57142857142857, 175.42857142857142), 
        (54.5, 155.0), (52.0, 147.0), (49.5, 139.0), (44.5, 123.5), (41.5, 115.5), (36.5, 100.5), (30.0, 80.5), (28.0, 74.5), (25.5, 66.5), 
        (23.5, 60.0), (20.5, 51.5), (18.5, 44.5), (11.5, 24.5), (5.5, 5.5)
    ], 
    [
        (216.0, 585.5), (215.5, 582.5), (214.5, 580.5), (213.5, 578.0), (212.8, 577.3), (212.0, 574.7142857142857), 
        (210.5, 571.5), (209.6, 569.4), (208.0, 565.3846153846154), (206.66666666666666, 561.8333333333334), (205.4, 558.6), (203.0, 552.5), 
        (201.91666666666666, 550.4166666666666), (200.46153846153845, 546.4615384615385), (198.78571428571428, 542.8571428571429), 
        (196.57142857142858, 537.9285714285714), (194.6, 533.4), (192.5, 527.5), (190.8, 524.1), (188.88888888888889, 519.2777777777778), 
        (186.6, 514.5), (184.5, 509.2857142857143), (182.64285714285714, 504.85714285714283), (180.0, 498.5), (177.71428571428572, 493.2857142857143), 
        (174.5, 485.5), (173.0, 482.0), (170.5, 475.0), (168.0, 469.5), (165.5, 463.5), (163.0, 457.5), (160.4, 451.6), 
        (157.33333333333334, 445.0), (155.0, 438.5), (152.33333333333334, 433.0), (149.5, 425.0), (146.33333333333334, 419.0), (143.5, 412.0), 
        (140.5, 404.5), (138.0, 399.0), (135.0, 391.5), (131.0, 382.5), (129.0, 377.5), (125.5, 369.0), (122.0, 362.0), (118.0, 353.0), 
        (115.0, 345.5), (112.0, 338.5), (109.0, 331.0), (105.5, 323.0), (102.5, 316.0), (98.5, 307.0), (95.0, 298.5), (91.5, 290.5), (88.5, 282.5), 
        (84.5, 273.0), (81.5, 266.0), (78.0, 256.0), (74.5, 248.0), (71.5, 239.0), (67.5, 230.0), (63.2, 218.8), (60.5, 211.5), (57.0, 202.0), 
        (53.0, 191.0), (50.0, 182.5), (50.23076923076923, 184.6153846153846), (48.45454545454545, 180.1818181818182), (44.0, 167.25), 
        (43.05555555555556, 165.5), (35.5, 145.5), (19.0, 86.0), (16.0, 73.5), (13.0, 62.333333333333336), (11.0, 54.5), (8.5, 43.5), (5.5, 32.0), (3.0, 21.0), (2.0, 10.5)
    ], 
    [
        (217.5, 585.5), (214.0, 577.0), (212.5, 573.5), (211.5, 570.0), (210.5, 567.5), (209.0, 564.0), (207.5, 560.5), 
        (206.0, 557.5), (205.0, 554.5), (203.5, 550.5), (201.5, 546.0), (200.0, 541.5), (198.0, 537.5), (195.0, 530.5), (192.0, 523.5), 
        (190.0, 519.0), (188.5, 516.0), (186.4, 510.9), (183.66666666666666, 504.0), (182.0, 499.5), (179.2, 493.9), (177.0, 487.5), 
        (174.66666666666666, 483.0), (172.0, 476.5), (169.66666666666666, 470.8333333333333), (167.33333333333334, 465.1666666666667), 
        (164.5, 458.0), (162.5, 452.5), (160.0, 446.0), (157.0, 440.0), (154.5, 433.0), (152.0, 427.0), (149.0, 420.0), (146.0, 412.5), 
        (143.5, 407.0), (141.0, 400.0), (138.0, 392.0), (135.0, 384.0), (132.0, 377.0), (129.5, 370.0), (126.5, 363.0), (123.5, 354.5), 
        (120.5, 347.0), (117.0, 338.0), (115.0, 332.0), (112.0, 324.0), (109.0, 315.5), (106.0, 306.5), (103.5, 299.5), (100.5, 291.5), 
        (97.0, 282.5), (94.0, 273.5), (95.66666666666667, 280.8888888888889), (90.66666666666667, 264.3333333333333), (89.5, 262.0), 
        (83.0, 238.5), (80.6, 231.0), (78.0, 221.5), (75.16666666666667, 212.33333333333334), (73.5, 206.0), (70.0, 195.0), (67.0, 185.5), 
        (66.33333333333333, 185.66666666666666), (62.0, 167.0), (60.5, 161.0), (57.0, 147.0), (55.0, 139.0), (52.5, 129.5), (50.0, 119.0), 
        (47.5, 109.0), (45.5, 99.5), (42.5, 89.0), (40.0, 80.0), (38.0, 71.0), (36.0, 62.0), (34.0, 52.5), (31.5, 41.0), (29.5, 31.5), 
        (27.5, 22.0), (25.5, 12.0), (23.5, 3.0)
    ], 

    [
        (644.5, 627.8518518518518), (644.0095238095238, 624.3904761904762), 
        (643.04, 618.6133333333333), (641.7435897435897, 612.6666666666666), (640.8, 607.0266666666666), (639.3882352941176, 601.2235294117647), 
        (638.5866666666667, 595.5866666666667), (637.283950617284, 589.3086419753087), (636.126582278481, 583.2658227848101), (635.025, 576.8), 
        (634.08, 570.7466666666667), (632.8625, 564.1625)
    ], 

    [
        (1090.0, 781.0), (1091.8620689655172, 776.3407707910751), (1088.695238095238, 777.2857142857143), 
        (1087.3738317757009, 779.4018691588785), (1086.6274509803923, 774.6013071895425), (1086.407643312102, 767.7261146496816), 
        (1085.6470588235295, 761.1), (1084.8657718120805, 753.6442953020135), (1084.2777777777778, 746.2530864197531), 
        (1083.0434782608695, 738.3260869565217), (1082.5414364640883, 730.3922651933701), (1081.839378238342, 722.699481865285), 
        (1080.7358490566037, 714.3820754716982), (1079.9550561797753, 705.9831460674158), (1078.8829268292684, 697.3121951219512), 
        (1077.9375, 688.7833333333333), (1076.8155339805826, 679.3592233009708), (1076.0348258706467, 670.4925373134329), 
        (1074.7543859649122, 661.640350877193), (1073.726368159204, 651.318407960199), (1072.4864864864865, 641.8648648648649), 
        (1071.2472727272727, 631.36), (1069.6728624535317, 621.9591078066915), (1068.25, 611.6610169491526), (1066.305439330544, 601.18410041841), 
        (1064.7509025270758, 589.9097472924187), (1062.9809160305344, 579.1335877862596), (1060.8939393939395, 567.8712121212121), 
        (1058.9059561128527, 556.1473354231975), (1056.8853046594982, 544.9283154121864), (1054.5255972696245, 533.2935153583618), 
        (1051.831081081081, 521.2533783783783), (1049.4489164086688, 508.8266253869969), (1046.8459119496856, 496.9811320754717), 
        (1043.926966292135, 484.5252808988764), (1041.0831234256927, 471.8942065491184), (1037.9967637540453, 459.453074433657), 
        (1034.7229916897506, 446.1246537396122), (1031.4735294117647, 433.0764705882353), (1027.9240837696334, 419.63612565445027), 
        (1024.419642857143, 406.3065476190476), (1020.6492753623188, 392.78840579710146), (1016.8605263157895, 378.5), 
        (1012.9373368146214, 364.9268929503917), (1008.8886138613861, 350.6039603960396), (1004.819587628866, 336.0180412371134), 
        (1000.3288043478261, 321.9320652173913), (995.8348416289592, 307.84615384615387), (991.2736842105263, 292.9978947368421), 
        (986.0796915167095, 277.5784061696658), (981.4009433962265, 263.45990566037733), (976.1860465116279, 248.10232558139535), 
        (970.726582278481, 232.4025316455696), (965.318, 217.832), (959.7690417690418, 202.33415233415235), (954.0025706940874, 186.8508997429306), 
        (948.4070450097847, 172.28571428571428), (942.1153846153846, 155.48076923076923), (935.8106796116505, 140.30582524271844), 
        (929.75226039783, 125.46112115732369), (923.3127659574468, 109.88936170212766), (916.839248434238, 94.29436325678498), 
        (910.1777777777778, 78.55757575757576), (903.1278538812785, 62.906392694063925), (896.5043478260869, 47.21739130434783), 
        (889.2724137931034, 31.662068965517243), (882.0761904761905, 15.80952380952381), (876.3303167420814, 6.276018099547511)
    ], 

    [
        (620.8582677165355, 447.86614173228344), (620.6153846153846, 441.3333333333333), (620.4710743801653, 433.35537190082647), 
        (620.5, 425.175), (620.3760683760684, 417.3931623931624), (620.3387096774194, 408.86290322580646), (620.2926829268292, 400.4959349593496), 
        (620.5, 392.6666666666667), (620.9684684684685, 384.73423423423424), (621.0, 375.384), (621.6711409395973, 366.5973154362416), 
        (622.2121212121212, 358.0378787878788), (622.9933774834437, 349.36423841059604), (623.725, 340.65625), 
        (624.4630872483222, 330.97986577181206), (625.8363636363637, 322.6848484848485), (626.8843930635838, 313.01734104046244), 
        (628.0982142857143, 304.5267857142857), (629.5364238410596, 295.07284768211923), (631.2388059701492, 286.25373134328356), 
        (632.7597765363129, 277.31284916201116), (635.1366120218579, 267.2295081967213), (637.3354838709678, 258.2), 
        (639.5318181818182, 248.23636363636365), (641.78, 238.73), (644.3288288288288, 229.5900900900901), (646.8910891089109, 220.3861386138614), 
        (649.8446215139443, 210.47011952191235), (652.9458128078818, 200.95566502463055), (655.9590163934427, 191.56967213114754), 
        (659.5196850393701, 181.34251968503938), (662.9422222222222, 172.1688888888889), (666.5092592592592, 162.52777777777777), 
        (670.497512437811, 152.73134328358208), (674.0960960960961, 143.24324324324323), (678.3657407407408, 133.02777777777777), 
        (682.6510791366907, 123.09352517985612), (687.0845921450151, 112.92749244712991), (691.3933933933934, 103.18318318318319), 
        (696.2332268370607, 92.45367412140575), (701.2208955223881, 81.48358208955224), (705.8763736263736, 71.57417582417582), 
        (711.1895043731779, 60.86880466472303), (716.3935309973045, 49.9622641509434), (722.0909090909091, 38.36363636363637), 
        (727.478431372549, 27.36078431372549), (733.5409836065573, 15.594262295081966), (739.094674556213, 6.011834319526627), 
        (749.0384615384615, 2.980769230769231)
    ], 

    [
        (518.0, 498.5), (519.0, 494.5), (519.5, 493.0), (520.4, 491.2), (521.25, 490.0), 
        (521.6666666666666, 488.3333333333333), (522.5, 486.5), (523.2, 484.6), (523.5, 483.5), (524.5, 481.5), 
        (525.3333333333334, 479.6666666666667), (526.0, 478.0), (526.5, 476.5), (527.5, 474.5), (528.0, 472.75), (528.5, 471.5), (529.5, 469.5), 
        (530.5, 467.5), (530.6666666666666, 466.3333333333333), (531.5, 464.5), (532.25, 463.0), (533.0, 461.25), (533.5, 459.5), (534.5, 457.5), 
        (535.3333333333334, 456.3333333333333), (535.6666666666666, 454.6666666666667), (536.3333333333334, 452.6666666666667), (537.5, 451.5), 
        (537.6666666666666, 450.3333333333333), (538.5, 448.5), (539.0, 447.0), (539.8, 445.4), (540.5, 444.5), 
        (540.6666666666666, 442.6666666666667), (541.5, 441.5), (542.3333333333334, 439.6666666666667), (543.3333333333334, 438.3333333333333), 
        (543.8, 436.6), (544.5, 435.5), (546.0, 434.5), (546.5, 433.5), (546.3333333333334, 431.6666666666667), 
        (546.6666666666666, 430.3333333333333), (548.0, 429.0), (549.0, 426.5), (550.5, 425.5), (549.5, 424.5), (550.1565557729941, 424.692759295499)
    ], 

    [
        (445.5, 498.5), (447.1666666666667, 497.1666666666667), (448.8, 495.6), (450.7142857142857, 494.0), (452.875, 492.125), 
        (454.5, 490.5), (456.2, 489.4), (458.0, 487.5), (460.0, 486.0), (462.0, 484.0), (463.8, 482.6), (465.8, 480.6), (467.5, 479.5), 
        (469.5, 477.5), (471.1666666666667, 476.1666666666667), (473.0, 474.5), (475.0, 472.75), (476.8, 471.4), (479.0, 469.5), (481.5, 468.0), 
        (482.5, 466.5), (485.0, 465.0), (486.0, 463.5), (488.5, 461.5), (490.1666666666667, 460.1666666666667), (492.0, 458.5), 
        (494.0, 456.7142857142857), (496.5, 455.5), (499.0, 454.0), (500.0, 452.5), (503.0, 451.5), (504.0, 449.5), (506.5, 448.5), (507.5, 446.5), 
        (510.0, 445.0), (512.0, 444.0), (514.0, 443.0), (514.6666666666666, 441.3333333333333), (516.5, 439.5), (518.5, 438.5), 
        (520.3333333333334, 436.6666666666667), (522.0, 435.75), (523.6666666666666, 434.3333333333333), (526.5, 433.5), (528.5, 432.0), 
        (529.5, 430.5), (532.0, 429.5), (534.0, 428.5), (534.8, 426.6), (536.5, 425.5), (537.5, 424.5), (539.5, 423.5), 
        (538.7454047914085, 425.63806278397357)
    ], 

    [
        (101.0, 141.0), (100.0, 139.0), (104.5, 136.5), (106.5, 133.5), (106.0, 128.5), (108.0, 132.0), (102.5, 123.0)
    ], 

    [
        (331.53333333333336, 397.1333333333333), (331.7857142857143, 396.7857142857143), 
        (339.6666666666667, 381.0833333333333), (339.85714285714283, 380.57142857142856), (345.0, 362.5), (351.0, 345.5)
    ], 

    [
        (532.0, 389.0), (542.7142857142857, 376.5), (544.7391304347826, 376.82608695652175), (543.3333333333334, 376.5925925925926), 
        (546.375, 376.5625), (541.7142857142857, 376.5), (544.9428571428572, 376.4), (542.6470588235294, 376.94117647058823), 
        (547.3571428571429, 376.2857142857143), (545.3, 376.4), (544.6756756756756, 376.4054054054054), (545.125, 376.25), 
        (550.0, 362.38461538461536), (543.1428571428571, 376.07142857142856), (558.0, 364.75), (563.5, 376.0), (559.0, 355.5), 
        (558.0, 366.2), (558.0714285714286, 366.42857142857144), (541.5, 369.5)
    ], 

    [
        (323.38709677419354, 368.3225806451613), (324.27777777777777, 367.52777777777777), (324.8048780487805, 367.8292682926829), 
        (324.5681818181818, 367.8636363636364), (327.05882352941177, 368.05882352941177), (324.69767441860466, 367.86046511627904), 
        (324.56756756756755, 367.81081081081084), (334.3333333333333, 368.3333333333333), (317.0, 370.2857142857143), (327.125, 371.875), 
        (334.4, 367.8), (324.72093023255815, 367.69767441860466), (323.27272727272725, 366.72727272727275), (333.5, 365.5), 
        (328.22222222222223, 362.0), (336.5, 347.5), (339.3333333333333, 345.6666666666667), (339.5, 345.5), (340.0, 349.5), (340.0, 346.5), 
        (339.5, 345.5), (339.5, 345.5), (342.0, 347.0), (339.5, 346.5), (339.5, 345.5), (339.8, 345.6), (339.6666666666667, 345.3333333333333), 
        (340.0, 347.0), (340.0, 346.5), (341.0, 343.5), (341.0, 346.5), (360.0, 344.5), (339.5, 346.0), (339.5, 345.5)
    ], 

    [
        (622.6666666666666, 371.73333333333335), (611.9090909090909, 367.6363636363636), (609.7906976744187, 366.4186046511628), 
        (610.2307692307693, 365.0), (606.1666666666666, 355.1666666666667), (609.7560975609756, 366.219512195122), 
        (609.7894736842105, 366.2105263157895), (608.0, 368.5), (611.3636363636364, 367.90909090909093), (605.0, 376.0), 
        (619.7857142857143, 378.35714285714283)
    ], 

    [
        (434.0625, 385.75), (432.5, 385.5), (422.92857142857144, 376.85714285714283), 
        (418.75, 363.6666666666667), (408.0, 348.5)
    ],  
        
    [
        (588.3333333333334, 374.53333333333336), (588.6153846153846, 374.38461538461536), 
        (588.5, 374.2857142857143), (585.0, 364.5), (590.0, 358.5), (590.1666666666666, 358.3333333333333), (597.5, 352.5)
    ],

    [
        (726.5, 294.5), (731.5, 276.5), (718.0, 270.0), (715.0, 267.0), (706.0, 260.0), (699.0, 258.0)
    ], 

    [
        (604.8333333333334, 376.0833333333333), (594.0, 375.75), (588.2, 374.5), (588.5, 374.2857142857143), (585.0, 364.5), 
        (588.6153846153846, 374.38461538461536), (588.6, 374.6), (585.0, 364.6666666666667)
    ], 

    [
        (429.2, 359.5), (434.5, 353.5), (438.5, 348.5), (428.57142857142856, 359.7142857142857), (418.7142857142857, 364.42857142857144), 
        (419.0, 363.5), (413.5, 362.5), (413.5, 362.2857142857143), (408.1395348837209, 361.43023255813955), (404.1666666666667, 360.6666666666667), 
        (403.22222222222223, 362.44444444444446), (408.17721518987344, 361.45569620253167), (414.57142857142856, 362.57142857142856), 
        (429.5, 365.5), (422.875, 376.125), (429.4, 359.4), (429.5, 359.5), (429.4375, 359.25), (431.5, 366.5), (424.5, 348.0), (436.5, 350.5)
    ], 

    [
        (349.5, 364.5), (361.0, 362.5), (372.89473684210526, 361.7368421052632), (389.35714285714283, 360.2142857142857), 
        (406.0, 346.5), (430.5, 351.0), (428.5, 359.5), (428.5, 366.5), (445.0, 356.0)
    ], 

    [
        (534.0, 318.5), (538.5, 312.5), (561.5, 310.5), (595.5, 298.5), (603.5, 295.0)
    ]
]

# ==========================================
# 1. Spatial Transformer
# ==========================================


def rotation_matrix(cos_sin):
    c, s = cos_sin[:, 0], cos_sin[:, 1]
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


class LocalizationNet(nn.Module):
    """Reads a (padded) trajectory and predicts a pose correction:
    shift (2), log-size (1), rotation as cos/sin (2).
    Each point is embedded with its relative position along the path, then
    pooled over the real (unpadded) points only."""

    def __init__(self, hidden=64):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()
        )
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 5))
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(self, x, mask, tau):
        h = self.point_mlp(torch.cat([x, tau], -1))   # (B, T, H)
        m = mask[..., :1]
        mean = (h * m).sum(1) / m.sum(1)
        mx = h.masked_fill(m == 0, -1e9).amax(1)
        out = self.head(torch.cat([mean, mx], -1))
        shift = out[:, :2]
        log_s = 2.0 * torch.tanh(out[:, 2])           # at most ~7x zoom per pass
        cos_sin = F.normalize(out[:, 3:5], dim=1)
        return shift, log_s, cos_sin


class SpatialTransformer(nn.Module):
    """Iterative spatial transformer: several passes, each refining the pose.
    Returns the canonical trajectory and the total learned pose (t, s, R) with
    raw ~= s * canonical @ R^T + t."""

    def __init__(self, passes=ST_PASSES):
        super().__init__()
        self.passes = passes
        self.loc = LocalizationNet()

    def forward(self, x, mask, tau):
        B = x.shape[0]
        t = torch.zeros(B, 1, 2)
        s = torch.ones(B)
        R = torch.eye(2).expand(B, 2, 2)
        canon = x
        for _ in range(self.passes):
            dt, dlog_s, dcs = self.loc(canon, mask, tau)
            dR, ds = rotation_matrix(dcs), torch.exp(dlog_s)
            canon = ((canon - dt[:, None]) @ dR) / ds[:, None, None] * mask
            t = t + s[:, None, None] * (dt[:, None] @ R.transpose(1, 2))
            R = R @ dR
            s = s * ds
        return canon, t, s, R


# ==========================================
# 2. Your LSTM Autoencoder, with the Spatial Transformer in front
# ==========================================


class STLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=2, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.st = SpatialTransformer()
        self.encoder_lstm = nn.LSTM(input_dim, latent_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(latent_dim, latent_dim, batch_first=True)
        # Tanh keeps the canonical drawing inside a fixed box. Real size, position
        # and rotation therefore have nowhere to go except the learned pose.
        self.output_layer = nn.Sequential(nn.Linear(latent_dim, input_dim), nn.Tanh())

    def encode(self, canon, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            canon, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.encoder_lstm(packed)
        return hidden.squeeze(0)

    def forward(self, x, mask, tau, lengths):
        canon, t, s, R = self.st(x, mask, tau)
        latent = self.encode(canon, lengths)
        dec_in = latent.unsqueeze(1).repeat(1, x.shape[1], 1)
        dec_out, _ = self.decoder_lstm(dec_in)
        recon_canon = self.output_layer(dec_out)
        recon = (s[:, None, None] * (recon_canon @ R.transpose(1, 2)) + t) * mask
        return recon, latent, canon, recon_canon


# ==========================================
# 3. Data
# ==========================================

# Same preprocessing as your version: start every track at (0, 0) and divide ALL
# tracks by ONE shared number. Relative size and rotation are left untouched.
def evenly_spaced(traj, n=RESAMPLE_POINTS):
    """Input formatting: n points evenly spaced along the path, so the LSTM can't
    tell a 5-point fragment from an 85-point track just by counting steps."""
    p = np.asarray(traj, np.float64)
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    if d[-1] == 0:
        return np.repeat(p[:1], n, 0).astype(np.float32)
    u = np.linspace(0, d[-1], n)
    return np.stack([np.interp(u, d, p[:, 0]), np.interp(u, d, p[:, 1])], 1).astype(np.float32)


raw_points = [len(t) for t in raw_trajectory_input]
resampled = [evenly_spaced(t) for t in raw_trajectory_input]
resampled_lengths = [len(t) for t in resampled]
NUM = len(resampled_lengths)
origin_tracks = [t - t[0] for t in resampled]
# One shared number = the typical (median) trajectory spread, so values sit
# around 1 instead of ~0.01. Big tracks stay big, small ones stay small.
global_scale = float(np.median([np.sqrt(((t - t.mean(0)) ** 2).sum(1).mean()) for t in origin_tracks]))
tracks = [torch.tensor(t / global_scale) for t in origin_tracks]
# the same tracks walked backwards (also starting at 0,0), for direction changes
tracks_rev = [(t.flip(0) - t.flip(0)[0]) for t in tracks]

X = nn.utils.rnn.pad_sequence(tracks, batch_first=True)
X_rev = nn.utils.rnn.pad_sequence(tracks_rev, batch_first=True)
T = X.shape[1]
MASK = torch.zeros_like(X)
TAU = torch.zeros(NUM, T, 1)
for i, l in enumerate(resampled_lengths):
    MASK[i, :l] = 1.0
    TAU[i, :l, 0] = torch.linspace(0, 1, l)
LEN = torch.tensor(resampled_lengths)


def random_pose(x, mask):
    """Random rotation, size (x0.33..x3) and position. Creates extra examples
    only; no labels."""
    B = x.shape[0]
    a = torch.rand(B) * 2 * math.pi
    R = rotation_matrix(torch.stack([torch.cos(a), torch.sin(a)], 1))
    k = torch.exp(torch.empty(B).uniform_(math.log(1 / 3), math.log(3)))
    shift = torch.empty(B, 1, 2).uniform_(-0.5, 0.5)
    return ((x @ R.transpose(1, 2)) * k[:, None, None] + shift) * mask


def masked_spread(x, mask):
    n = mask[..., 0].sum(1)
    c = (x * mask).sum(1, keepdim=True) / n[:, None, None]
    return (((x - c) ** 2) * mask).sum((1, 2)) / n


def size_fair_error(recon, x, mask):
    """Error relative to the trajectory's own spread, so a 10-px track counts as
    much as a 600-px one. Only weights the loss; the model never sees it."""
    n = mask[..., 0].sum(1)
    err = (((recon - x) ** 2) * mask).sum((1, 2)) / n
    return err / (masked_spread(x, mask).detach() + 1e-10)


def batch_of_copies():
    idx = torch.arange(NUM).repeat(COPIES_PER_STEP)
    rev = torch.rand(len(idx)) < 0.5
    base = torch.where(rev[:, None, None], X_rev[idx], X[idx])
    m, tau, ln = MASK[idx], TAU[idx], LEN[idx]
    return random_pose(base, m), random_pose(base, m), m, tau, ln


# ==========================================
# 4. Training
# ==========================================

model = STLSTMAutoencoder()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS)

demo = random_pose(X, MASK)          # fixed random poses, filmed during training
snapshots = []

print(f"Training on {NUM} trajectories ...")
model.train()
for epoch in range(1, NUM_EPOCHS + 1):
    xa, xb, m, tau, ln = batch_of_copies()
    both = torch.cat([xa, xb])
    m2, tau2, ln2 = torch.cat([m, m]), torch.cat([tau, tau]), torch.cat([ln, ln])
    recon, latent, canon, _ = model(both, m2, tau2, ln2)

    rec_loss = torch.log1p(size_fair_error(recon, both, m2)).mean()
    ca, cb = canon.chunk(2)
    # canonical forms of two copies of the same trajectory should agree
    # (measured relative to their own size, so shrinking them doesn't help)
    agree = (((ca - cb) ** 2) * m).sum((1, 2)) / m[..., 0].sum(1)
    agree = agree / (masked_spread(ca, m) + masked_spread(cb, m) + 1e-10)
    # keep the canonical drawing at the size of the decoder's output box
    # (fixes the canvas size; the transformer still has to work out each track's size)
    canvas = (torch.log(masked_spread(canon, m2) + 1e-10) - math.log(CANVAS_SPREAD)) ** 2
    loss = rec_loss + CONSISTENCY_WEIGHT * agree.mean() + 0.1 * canvas.mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if epoch % SNAPSHOT_EVERY == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            _, _, c, _ = model(demo, MASK, TAU, LEN)
        model.train()
        snapshots.append((epoch, c.numpy().copy(), loss.item()))
    if epoch % 50 == 0:
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] | reconstruction {rec_loss.item():.4f}"
              f" | canonical disagreement {agree.mean().item():.4f}")

print("\nModel trained successfully!")


# ==========================================
# 5. Latent vectors, noise score, HDBSCAN
# ==========================================

model.eval()
K = 32
with torch.no_grad():
    recon, latent_vectors, canon, recon_canon = model(X, MASK, TAU, LEN)
    latent_vectors = latent_vectors.numpy()
    # invariance check: latents of random size/rotation copies
    lat_copies = torch.stack([model(random_pose(X, MASK), MASK, TAU, LEN)[1] for _ in range(K)])
    err = torch.stack([size_fair_error(model(xp, MASK, TAU, LEN)[0], xp, MASK)
                       for xp in (random_pose(X, MASK) for _ in range(K))]).mean(0).numpy()

within = (lat_copies - lat_copies.mean(0)).norm(dim=-1).mean().item()
pair = torch.cdist(torch.tensor(latent_vectors), torch.tensor(latent_vectors))
print(f"\nInvariance check: latent moves {within:.4f} when size/rotation change; "
      f"median distance between different trajectories {pair[pair > 0].median().item():.4f}")

# STEP 1 - Decides which trajectories are noise (can't be reconstructed) and which are shapes (can be reconstructed)
log_err = np.log(err + 1e-12)
order = np.sort(log_err)
best_cut, best_cost = None, np.inf
for k in range(1, NUM):
    lo, hi = order[:k], order[k:]
    cost = lo.var() * len(lo) + hi.var() * len(hi)
    if cost < best_cost:
        best_cost, best_cut = cost, (order[k - 1] + order[k]) / 2
lo_mean = log_err[log_err <= best_cut].mean()
hi_mean = log_err[log_err > best_cut].mean()
# only call it noise if the two groups really differ (> ~3x in error)
recon_noise = (log_err > best_cut) if hi_mean - lo_mean > math.log(3) else np.zeros(NUM, bool)
noise_threshold = math.exp(best_cut)

# STEP 2 - Takes the trajectories that can be reconstructed and groups them into shape clusters
final = np.full(NUM, -1)
strength = np.zeros(NUM)
shape_idx = np.where(~recon_noise)[0]
if len(shape_idx) >= 3:
    # don't split groups that are closer than 1/4 of the typical latent distance
    eps = 0.25 * float(np.median(pair[pair > 0]))
    clusterer = HDBSCAN(min_cluster_size=3, min_samples=2, allow_single_cluster=True,
                        cluster_selection_epsilon=eps, copy=True)
    final[shape_idx] = clusterer.fit_predict(latent_vectors[shape_idx])
    strength[shape_idx] = clusterer.probabilities_

# STEP 3 - group the noise trajectories among themselves (types of noise)
noise_idx = np.where(final == -1)[0]
noise_group = np.full(NUM, -1)
if len(noise_idx) >= 3:
    noise_group[noise_idx] = HDBSCAN(min_cluster_size=2, min_samples=1, copy=True
                                     ).fit_predict(latent_vectors[noise_idx])

print(f"\nNoise threshold learned from the model's errors: {noise_threshold:.3f}")
print("\nClustering Results:")
for i in range(NUM):
    if final[i] >= 0:
        name = f"Shape cluster {final[i]} | Membership strength: {strength[i]:.2f}"
    else:
        why = "can't be reconstructed" if recon_noise[i] else "no dense shape group"
        grp = f", noise group {noise_group[i]}" if noise_group[i] >= 0 else ""
        name = f"Noise ({why}{grp})"
    print(f"Trajectory #{i:2d} ({raw_points[i]:2d} points) -> {name} | Noise score: {err[i]:.3f}")

torch.save(model.state_dict(), os.path.join(OUT_DIR, "st_lstm_autoencoder.pt"))


# ==========================================
# 6. Figures: results + training simulation GIF
# ==========================================

palette = ["#2a7de1", "#e8743b", "#19a979", "#945ecf", "#d64e8c", "#13a4b4"]
colour = lambda l: "#9a9a9a" if l < 0 else palette[l % len(palette)]
label = lambda l: "noise" if l < 0 else f"cluster {l}"

cols = 5
rows = math.ceil(NUM / cols)
fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 3, rows * 5.2))
for i in range(NUM):
    r, c = divmod(i, cols)
    a1, a2 = axes[2 * r, c], axes[2 * r + 1, c]
    l = resampled_lengths[i]
    raw = np.asarray(raw_trajectory_input[i])
    a1.plot(raw[:, 0], raw[:, 1], color=colour(final[i]), lw=1.6)
    a1.plot(*raw[0], "o", color=colour(final[i]), ms=3)
    a1.set_title(f"#{i} raw - {label(final[i])}", fontsize=8)
    cn, rc = canon[i, :l].numpy(), recon_canon[i, :l].numpy()
    a2.plot(cn[:, 0], cn[:, 1], color=colour(final[i]), lw=1.6)
    a2.plot(rc[:, 0], rc[:, 1], "--", color="k", lw=.8)
    a2.set_title(f"#{i} after spatial transformer", fontsize=8)
    for a in (a1, a2):
        a.set_aspect("equal", "datalim"); a.set_xticks([]); a.set_yticks([])
for j in range(NUM, rows * cols):
    r, c = divmod(j, cols)
    axes[2 * r, c].axis("off"); axes[2 * r + 1, c].axis("off")
fig.suptitle("Top: raw trajectory. Bottom: canonical form learned by the spatial "
             "transformer (dashed = decoder reconstruction)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(os.path.join(OUT_DIR, "results.png"), dpi=120)
plt.close(fig)

fig, axes = plt.subplots(2, math.ceil(NUM / 2), figsize=(18, 4.4))
axes = axes.ravel()
def draw(k):
    epoch, c, l = snapshots[k]
    for j, a in enumerate(axes):
        a.cla(); a.set_xticks([]); a.set_yticks([])
        if j >= NUM:
            a.axis("off"); continue
        p = c[j, :resampled_lengths[j]]
        a.plot(p[:, 0], p[:, 1], color=colour(final[j]), lw=1.6)
        a.plot(p[0, 0], p[0, 1], "o", color=colour(final[j]), ms=3)
        a.set_title(f"#{j}", fontsize=8)
        a.set_aspect("equal", "datalim")
    fig.suptitle(f"Spatial transformer output during training - epoch {epoch}, loss {l:.3f}")
FuncAnimation(fig, draw, frames=len(snapshots)).save(
    os.path.join(OUT_DIR, "training_simulation.gif"), writer=PillowWriter(fps=4))
plt.close(fig)
print(f"\nSaved {OUT_DIR}/results.png and {OUT_DIR}/training_simulation.gif")
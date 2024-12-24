import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
imgL = cv.imread('./data/left.png', cv.IMREAD_GRAYSCALE)
imgR = cv.imread('./data/right.png', cv.IMREAD_GRAYSCALE)
 

 
stereo = cv.StereoBM.create(numDisparities=64, blockSize=21) # line 0~64    


disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()



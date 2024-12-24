import cv2
import numpy as np 
from copy import deepcopy

#crop (cutting image)
def crop_quarter(image, height, width):
    return image[0:int(height/2), 0:int(width/2)] # 0qnxj height half! cut!!    +   0~ width half cup
# but, x , yrk dkslfk!!!!! (y,x!!!)<- important


def resize_to_original(image, original_height, original_width):
    return cv2.resize(image, (original_width, original_height))


def add_gaussian(image, height, width):
    gaussian_noise = np.random.normal(0.1, 0.5, (height, width, 3)) #가우시안 노이즈 생성
    result = image + gaussian_noise #노이즈 더하기
    result = result.astype(np.uint8) #노이즈가 실수형이므로, 정수형으로 변환
    result = np.clip(result, 0, 255) #이미지는 0~255사이의 값을 가져야하므로, 범위를 넘어서는 값 처리
    return result   


def main(args=None):
    file_name="camera.png"
    img=cv2.imread(file_name)
    
    original_image = deepcopy(img) # 원본이미지보관
    cv2.imshow("original", img)
    

    #image ku gi(size information)
    print(img.shape)
    # 이미지 1/4로 자르기
    height, width, _ = img.shape 

    #image 1/4
    img = crop_quarter(img, height, width)
    #cv2.imshow("crop", img)
   # cv2.waitKey(0)

    img = resize_to_original(img, height, width)
    #cv2.imshow("zoom", img)
 #   cv2.waitKey(0)


    img = add_gaussian(original_image, height, width)
    #cv2.imshow("Gaussian", img)
   # cv2.waitKey(0)


# edge 필터
    cropped_image = original_image[int(height/2):height, :] #ROI  (half height ~ height cut!!) -> road line perception
    canny_image = cv2.Canny(cropped_image, 100, 200, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(canny_image, 1, np.pi/180, 140, minLineLength=1, maxLineGap=10)
    hough_image = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for i in range(lines.shape[0]):
            pt1 = (lines[i][0][0], lines[i][0][1]) #시작점 좌표
            pt2 = (lines[i][0][2], lines[i][0][3]) #끝점 좌표
            cv2.line(hough_image, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA) #line draw function (red Line!!)

    cv2.imshow("line detection", hough_image)
    cv2.waitKey(0)

   


    #cv2.waitKey(0) 
    return 0
    

if __name__== '__main__':
    main()

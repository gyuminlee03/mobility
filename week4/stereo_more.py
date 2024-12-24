import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 왼쪽 및 오른쪽 이미지를 그레이스케일로 읽어옵니다.
imgL = cv.imread('./data/left.png', cv.IMREAD_GRAYSCALE)
imgR = cv.imread('./data/right.png', cv.IMREAD_GRAYSCALE)

# 이미지가 정상적으로 읽혔는지 확인합니다.
if imgL is None or imgR is None:
    print("Error: One of the images did not load. Check the file paths.")
    exit()

# 이미지 크기를 가져옵니다.
height, width = imgL.shape

# 최대 Disparity 범위
max_disparity = 64

# Disparity 맵을 저장할 배열을 생성합니다.
disparity = np.zeros((height, width), dtype=np.float32)

# SSD (Sum of Squared Differences) 함수를 정의합니다.
def ssd(I1, I2, x, y, d, block_size):
    half_block = block_size // 2
    ssd_value = 0

    # 블록의 경계를 확인하여 유효성 검사
    for i in range(-half_block, half_block + 1):
        for j in range(-half_block, half_block + 1):
            # 경계 조건을 체크합니다.
            if (0 <= y + j < height) and (0 <= x + i < width) and (0 <= x + i - d < width):
                diff = int(I1[y + j, x + i]) - int(I2[y + j, x + i - d])
                ssd_value += diff ** 2  # 제곱 차이를 추가합니다.

    return ssd_value

# 각 픽셀에 대해 Disparity를 계산합니다.
for y in range(height):
    for x in range(width):
        best_disparity = 0
        best_ssd = float('inf')

        # Disparity 값 0부터 max_disparity까지 반복합니다.
        for d in range(max_disparity):
            # SSD를 계산합니다.
            ssd_value = ssd(imgL, imgR, x, y, d, block_size=5)

            # 현재 SSD 값이 가장 낮은 경우를 찾습니다.
            if ssd_value < best_ssd:
                best_ssd = ssd_value
                best_disparity = d

        # 최적의 Disparity 값을 저장합니다.
        disparity[y, x] = best_disparity

# Disparity 맵을 정규화합니다.
disparity = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

# Disparity 맵을 컬러맵으로 변환합니다.
disparity_colormap = cv.applyColorMap(disparity, cv.COLORMAP_JET)

# 결과를 출력합니다.
plt.imshow(disparity_colormap)
plt.title('Disparity Map')
plt.axis('off')
plt.show()


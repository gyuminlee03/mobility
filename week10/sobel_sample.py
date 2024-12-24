from torch import nn
import torch
import cv2
from matplotlib import pyplot as plt
import time
from glob import glob
import numpy as np

class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d( in_channels=1, out_channels=2,
                                kernel_size=3,stride=1,padding=0,bias=False )

        Gx = torch.tensor([[2.0,0.0,-2.0],[4.0,0.0,-4.0], [2.0,0.0,-2.0]]) #3*3
        Gy = torch.tensor([[2.0,4.0,2.0],[0.0,0.0,0.0], [-2.0,-4.0,-2.0]]) #3*3
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0) #2*3*3
        G = G.unsqueeze(1) #2*1*3*3

        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):

        # torch.mul(tensor1, tensor2): 두 tensor의 곱
        # torch.sum(tensor, dim=?): dim 방향으로 tensor를 합
        # torch.sqrt(tensor): tensor의 원소를 제곱


        x = self.filter(img) # batch * channel * h * w
        x = torch.mul(x,x)
        x = torch.sum(x,dim=1,keepdim=True)
        x = torch.sqrt(x)

        return x
    

def np_img_to_tensor(img):
    if len(img.shape) == 2:
        img = img[..., np.newaxis] # 새로운 차원추가 conv2d에서는 (batch, c, h, w)의 tensor를 입력으로 받음
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.permute(2, 0, 1) # 끝에 추가되었던 batch 차원을 앞으로 이동
    img_tensor = torch.unsqueeze(img_tensor, 0)
    return img_tensor


def tensor_to_np_img(img_tensor):
    img = img_tensor.cpu().permute(0, 2, 3, 1).numpy()
    return img[0, ...]


def sobel_torch_version(img_np, torch_sobel):
    img_tensor = np_img_to_tensor(np.float32(img_np))  # numpy array에서 로 타입 변경
    img_edged = tensor_to_np_img(torch_sobel(img_tensor))  # torch tensor에서 numpy array로 타입 변경
    img_edged = np.squeeze(img_edged)  # 크기 1의 차원을 제거
    return img_edged


def main():
    img_path = "./lenna.jpg"
    torch_sobel = Sobel()

    rgb_orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    rgb_orig = cv2.resize(rgb_orig, (224, 224))

    # torch 버전 sobel 필터
    rgb_edged = sobel_torch_version(rgb_orig, torch_sobel=torch_sobel)
    
    # cv2 버전 sobel 필터
    rgb_edged_cv2_x = cv2.Sobel(rgb_orig, cv2.CV_64F, 1, 0, ksize=3)
    rgb_edged_cv2_y = cv2.Sobel(rgb_orig, cv2.CV_64F, 0, 1, ksize=3)
    rgb_edged_cv2 = np.sqrt(np.square(rgb_edged_cv2_x), np.square(rgb_edged_cv2_y))

    rgb_orig = cv2.resize(rgb_orig, (222, 222))
    rgb_edged_cv2 = cv2.resize(rgb_edged_cv2, (222, 222))
    rgb_both = np.concatenate(
        [rgb_orig / 255, rgb_edged / np.max(rgb_edged), rgb_edged_cv2 / np.max(rgb_edged_cv2)], axis=1)

    plt.imshow(rgb_both, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
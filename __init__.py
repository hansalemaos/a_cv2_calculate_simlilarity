import os
import re
from typing import Any

import numpy as np
import cv2
import requests
import PILasOPENCV


def getMSSISM(i1, i2):
    # from opencv page
    C1 = 6.5025
    C2 = 58.5225
    # INITS
    I1 = np.float32(i1)  # cannot calculate on one byte large values
    I2 = np.float32(i2)
    I2_2 = I2 * I2  # I2^2
    I1_2 = I1 * I1  # I1^2
    I1_I2 = I1 * I2  # I1 * I2
    # END INITS
    # PRELIMINARY COMPUTING
    mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)
    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_2 = cv2.GaussianBlur(I1_2, (11, 11), 1.5)
    sigma1_2 -= mu1_2
    sigma2_2 = cv2.GaussianBlur(I2_2, (11, 11), 1.5)
    sigma2_2 -= mu2_2
    sigma12 = cv2.GaussianBlur(I1_I2, (11, 11), 1.5)
    sigma12 -= mu1_mu2
    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2  # t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2  # t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    ssim_map = cv2.divide(t3, t1)  # ssim_map =  t3./t1;
    mssim = cv2.mean(ssim_map)  # mssim = average of ssim map
    return mssim


def open_image_in_cv(image, channels_in_output=None):
    if isinstance(image, str):
        if os.path.exists(image):
            if os.path.isfile(image):
                image = cv2.imread(image)
        elif re.search(r"^.{1,10}://", str(image)) is not None:
            x = requests.get(image).content
            image = cv2.imdecode(np.frombuffer(x, np.uint8), cv2.IMREAD_COLOR)
        else:
            image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    elif "PIL" in str(type(image)):
        image = np.array(image)
    else:
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

    if channels_in_output is not None:
        if image.shape[-1] == 4 and channels_in_output == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.shape[-1] == 3 and channels_in_output == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        else:
            pass
    return image


def calculate_simlilarity(im1: Any, im2: Any) -> tuple:
    r"""
    from a_cv2_calculate_simlilarity import add_similarity_to_cv2
    add_similarity_to_cv2() #monkeypatch

    #if you don't want to use a monkey patch:
    #from a_cv2_calculate_simlilarity import calculate_simlilarity

    xx=cv2.calculate_simlilarity_of_2_pics(r"https://avatars.githubusercontent.com/u/77182807?v=4", r"https://avatars.githubusercontent.com/u/77182807?v=4")
    yy=cv2.calculate_simlilarity_of_2_pics(r"https://avatars.githubusercontent.com/u/77182807?v=4", r'https://avatars.githubusercontent.com/u/1024025?v=4')

    xx
    Out[5]: (1.0, 1.0, 1.0, 0.0)
    yy
    Out[6]: (0.04821188521851477, 0.04095997109929877, 0.02169693238520613, 0.0)

        Parameters:
            im1:Any
                image1 as url,file path,base64,numpy,PIL
            im2:Any
                image1 as url,file path,base64,numpy,PIL
        Returns:
              tuple
    """

    firstimage = open_image_in_cv(image=im1, channels_in_output=3)
    secondimage = open_image_in_cv(image=im2, channels_in_output=3)
    firstimage = PILasOPENCV.fromarray(firstimage)
    secondimage = PILasOPENCV.fromarray(secondimage)
    firstimagesize = (100, int(firstimage.size[1] * (100 / firstimage.size[0])))
    firstimagereduced = firstimage.resize(firstimagesize)
    secondimagereduced = secondimage.resize(firstimagesize)
    return getMSSISM(firstimagereduced.getim(), secondimagereduced.getim())


def add_similarity_to_cv2():
    cv2.calculate_simlilarity_of_2_pics = calculate_simlilarity

from typing import Any
from allescacher import cache_everything, cache_dict_module
import numpy as np
import cv2
import numexpr
from a_cv_imwrite_imread_plus import open_image_in_cv
from collections import defaultdict
import pandas as pd

C1 = 6.5025
C2 = 58.5225


def getMSSISM(I1, sigma1_2x, mu1x, mu1_2, I2, sigma2_2x, mu2x, mu2_2):
    sigma1_2 = sigma1_2x
    sigma2_2 = sigma2_2x
    I1_I2 = numexpr.evaluate("I1 * I2", global_dict={}, local_dict={"I1": I1, "I2": I2})

    mu1 = mu1x
    mu2 = mu2x

    mu1_mu2 = numexpr.evaluate(
        "mu1 * mu2",
        global_dict={},
        local_dict={
            "mu1": mu1,
            "mu2": mu2,
        },
    )

    numexpr.evaluate(
        "sigma1_2 - mu1_2",
        global_dict={},
        local_dict={
            "sigma1_2": sigma1_2,
            "mu1_2": mu1_2,
        },
        out=sigma1_2,
    )

    numexpr.evaluate(
        "sigma2_2 - mu2_2",
        global_dict={},
        local_dict={
            "sigma2_2": sigma2_2,
            "mu2_2": mu2_2,
        },
        out=sigma2_2,
    )

    sigma12 = cv2.GaussianBlur(I1_I2, (11, 11), 1.5)

    numexpr.evaluate(
        "sigma12 - mu1_mu2",
        global_dict={},
        local_dict={
            "sigma12": sigma12,
            "mu1_mu2": mu1_mu2,
        },
        out=sigma12,
    )

    t1 = numexpr.evaluate(
        "2 * mu1_mu2 + C1",
        global_dict={},
        local_dict={
            "mu1_mu2": mu1_mu2,
            "C1": C1,
        },
    )

    t2 = numexpr.evaluate(
        "2 * sigma12 + C2",
        global_dict={},
        local_dict={
            "sigma12": sigma12,
            "C2": C2,
        },
    )

    t3 = numexpr.evaluate(
        "t1 * t2",
        global_dict={},
        local_dict={
            "t1": t1,
            "t2": t2,
        },
    )

    t1 = numexpr.evaluate(
        "mu1_2 + mu2_2 + C1",
        global_dict={},
        local_dict={"mu1_2": mu1_2, "mu2_2": mu2_2, "C1": C1},
    )

    t2 = numexpr.evaluate(
        "sigma1_2 + sigma2_2 + C2",
        global_dict={},
        local_dict={"sigma1_2": sigma1_2, "sigma2_2": sigma2_2, "C2": C2},
    )

    t1 = numexpr.evaluate(
        "t1 * t2",
        global_dict={},
        local_dict={
            "t1": t1,
            "t2": t2,
        },
    )

    ssim_map = numexpr.evaluate(
        "t3 / t1",
        global_dict={},
        local_dict={
            "t3": t3,
            "t1": t1,
        },
    )

    mssim = cv2.mean(ssim_map)
    return mssim[2], mssim[1], mssim[0], mssim[3]


@cache_everything
def open_with_cv2_cached(
    im, width, height, interpolation=cv2.INTER_LINEAR, with_alpha=False
):
    bi1 = cv2.resize(
        open_image_in_cv(image=im, channels_in_output=4 if with_alpha else 3),
        (width, height),
        interpolation=interpolation,
    ).astype(np.float64)
    squa = cv2.GaussianBlur(
        numexpr.evaluate(
            "bi1 * bi1",
            global_dict={},
            local_dict={
                "bi1": bi1,
            },
        ),
        (11, 11),
        1.5,
    ).astype(np.float64)

    mu1 = cv2.GaussianBlur(bi1, (11, 11), 1.5)

    mu1_2 = numexpr.evaluate(
        "mu1 * mu1",
        global_dict={},
        local_dict={
            "mu1": mu1,
        },
    )

    return bi1, squa, mu1, mu1_2


def calculate_simlilarity(
    im1: Any,
    im2: Any,
    width=100,
    height=100,
    interpolation=cv2.INTER_LINEAR,
    with_alpha=False,
) -> tuple:
    r"""
    Calculate structural similarity between two images.

    This function computes the structural similarity index (SSIM) between two images,
    which measures the similarity of their structural patterns. The SSIM values range
    from -1 to 1, where a higher value indicates greater similarity.

    Parameters:
        im1: Any
            Image 1, which can be provided as a URL, file path, base64 string, numpy array,
            or PIL image.
        im2: Any
            Image 2, which can be provided as a URL, file path, base64 string, numpy array,
            or PIL image.
        width: int, optional
            Width of the resized images for comparison (default is 100).
        height: int, optional
            Height of the resized images for comparison (default is 100).
        interpolation: int, optional
            Interpolation method for resizing (default is cv2.INTER_LINEAR).
        with_alpha: bool, optional
            Whether to include alpha channel if present (default is False).

    Returns:
        tuple
            A tuple containing four SSIM values in the order (B, G, R, Alpha).

    Example:
        resa = calculate_simlilarity(
            r"https://avatars.githubusercontent.com/u/77182807?v=4",
            r"https://avatars.githubusercontent.com/u/77182807?v=4",
            width=100,
            height=100,
            interpolation=cv2.INTER_LINEAR,
            with_alpha=False,
        )
        print(resa)
        resa2 = calculate_simlilarity(
            r"https://avatars.githubusercontent.com/u/77182807?v=4",
            r"https://avatars.githubusercontent.com/u/1024025?v=4",
            width=100,
            height=100,
            interpolation=cv2.INTER_LINEAR,
            with_alpha=False,
        )
        print(resa2)

        resa2 = calculate_simlilarity(
            r"C:\Users\hansc\Downloads\1633513733_526_Roblox-Royale-High.jpg",
            r"C:\Users\hansc\Downloads\Roblox-Royale-High-Bobbing-For-Apples (1).jpg",
            width=100,
            height=100,
            interpolation=cv2.INTER_LINEAR,
            with_alpha=False,
        )
        print(resa2)

        resa2 = calculate_simlilarity(
            r"C:\Users\hansc\Documents\test1.png",
            r"C:\Users\hansc\Documents\test2.png",
            width=100,
            height=100,
            interpolation=cv2.INTER_LINEAR,
            with_alpha=False,
        )
        print(resa2)

    """
    I1, sigma1_2, mu1, mu1_2 = open_with_cv2_cached(
        im1, width, height, interpolation=interpolation, with_alpha=with_alpha
    )

    I2, sigma2_2, mu2, mu2_2 = open_with_cv2_cached(
        im2, width, height, interpolation=interpolation, with_alpha=with_alpha
    )

    return getMSSISM(
        I1.copy(),
        sigma1_2.copy(),
        mu1.copy(),
        mu1_2.copy(),
        I2.copy(),
        sigma2_2.copy(),
        mu2.copy(),
        mu2_2.copy(),
    )


def compare_all_images_with_all_images(
    imagelist,
    width=100,
    height=100,
    interpolation=cv2.INTER_LINEAR,
    with_alpha=False,
    delete_cache=True,
):
    r"""
    Compare a list of images with each other and return a similarity matrix.

    This function compares a list of images with each other using the `calculate_simlilarity`
    function and returns a similarity matrix as a pandas DataFrame. Each element in the matrix
    represents the similarity between two images.

    Parameters:
        imagelist: list
            List of images to compare. Each image can be provided as a URL, file path, base64 string,
            numpy array, or PIL image.
        width: int, optional
            Width of the resized images for comparison (default is 100).
        height: int, optional
            Height of the resized images for comparison (default is 100).
        interpolation: int, optional
            Interpolation method for resizing (default is cv2.INTER_LINEAR).
        with_alpha: bool, optional
            Whether to include alpha channel if present (default is False).
        delete_cache: bool, optional
            Whether to clear the cache of preprocessed images (default is True).

    Returns:
        pandas.DataFrame
            A DataFrame representing the similarity matrix between the images.

    Example:
        add_similarity_to_cv2()
        df = cv2.calculate_simlilarity_of_all_pics(
            [
                r"C:\Users\hansc\Downloads\testcompare\10462.7191107_0.png",
                r"C:\Users\hansc\Downloads\testcompare\10462.7191107_1.png",
                r"C:\Users\hansc\Downloads\testcompare\10462.7213836_0.png",
                r"C:\Users\hansc\Downloads\testcompare\10462.7213836_1.png",
                r"C:\Users\hansc\Downloads\testcompare\10462.7253843_0.png",
                r"C:\Users\hansc\Downloads\testcompare\10462.7253843_1.png",
                r"C:\Users\hansc\Downloads\testcompare\10462.7274426_0.png",
                r"C:\Users\hansc\Downloads\testcompare\10462.7274426_1.png",
                r"C:\Users\hansc\Downloads\testcompare\10462.7286225_0.png",
                r"C:\Users\hansc\Downloads\testcompare\10462.7286225_1.png",
                r"C:\Users\hansc\Downloads\testcompare\10462.7301136_0.png",
                r"C:\Users\hansc\Downloads\testcompare\10462.7301136_1.png",
                r"C:\Users\hansc\Downloads\testcompare\10462.7312635_0.png",
                r"C:\Users\hansc\Downloads\testcompare\10462.7312635_1.png",
                r"C:\Users\hansc\Downloads\testcompare\10462.7325586_0.png",
            ],
            width=100,
            height=100,
            interpolation=cv2.INTER_LINEAR,
            with_alpha=False,
        )
    """
    d = defaultdict(list)
    for ini1, fi1 in enumerate(imagelist):
        for ini2, fi2 in enumerate(imagelist):
            resa = calculate_simlilarity(
                fi1,
                fi2,
                width=width,
                height=height,
                interpolation=interpolation,
                with_alpha=with_alpha,
            )
            d[ini1].append([resa])

    df = pd.DataFrame(d)
    df.index = df.columns.copy()
    if delete_cache:
        cache_dict_module.cache_dict[
            open_with_cv2_cached.__qualname__
        ].clear()
    return df.explode(df.columns.to_list())


def add_similarity_to_cv2():
    cv2.calculate_simlilarity_of_2_pics = calculate_simlilarity
    cv2.calculate_simlilarity_of_all_pics = compare_all_images_with_all_images





# Calculate the simlilarity of 2 or more pictures with OpenCV

## Tested against Windows 10 / Python 3.11 / Anaconda

### pip install a-cv2-calculate_simlilarity


```python

from a_cv2_calculate_simlilarity import add_similarity_to_cv2
add_similarity_to_cv2() #monkeypatch

#if you don't want to use a monkey patch:
#from a_cv2_calculate_simlilarity import calculate_simlilarity


calculate_simlilarity(
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
        
        
compare_all_images_with_all_images(
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
```
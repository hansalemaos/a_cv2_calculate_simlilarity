```python
$pip install a-cv2-calculate_simlilarity
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
```
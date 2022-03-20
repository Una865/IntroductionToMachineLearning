import numpy as np
import PIL
from PIL import Image
from PIL import ImageOps

'''
Very simple object counter for 1D images
'''

def num_edges(im1):
    f1 = np.array([-1,1])
    im1 = np.insert(im1, 0, 0)
    n = im1.shape[0]

    edges = np.zeros(n)
    for i in range(n-1):
        prod = im1[i]*(-1)+im1[i+1]
        if prod<0:
            prod = 0
        edges[i] = prod
    return len(np.argwhere(edges==1))


im1 = np.array([1,0,1,1,1,0,1,0,0,1,1,0,1])
print(num_edges(im1))
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# for real faces

# input_dir = './training_real/'

# output_dir = './scaled/'
# if not os.path.exists(output_dir):
#     os.mkdir(output_dir)

# output_dir_train = output_dir+'trainA/'
# output_dir_test = output_dir+'testA/'
# if not os.path.exists(output_dir_train):
#     os.mkdir(output_dir_train)
# if not os.path.exists(output_dir_test):
#     os.mkdir(output_dir_test)
    
# train_size = 850
    
def k_downscale(image, k):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).

    Returns
        downscaled_image: A one-third-downscaled version of image.
    """
    m, n, c = image.shape
    image_out = np.zeros((int(m/k),int(n/k),c))
    for i in range(0,m,k):
        for j in range(0,n,k):
            image_out[int(i/k),int(j/k),:] = image[i,j,:]
    return image_out
    

# for idx, img in enumerate(os.listdir(input_dir)):
#     new_img = cv2.imread(os.path.join(input_dir, img))[..., ::-1].astype(float) 
#     new_img /= new_img.max()
#     new_img = k_downscale(new_img, 3)
#     if idx < train_size:
#         plt.imsave(output_dir_train+img, new_img)
#     else:
#         plt.imsave(output_dir_test+img, new_img)
    
# print("downscaled img size:", new_img.shape)


# for cartoons

input_dir = './cartoon_training_real/'

output_dir = './scaled/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_dir_train = output_dir+'trainB/'
output_dir_test = output_dir+'testB/'
if not os.path.exists(output_dir_train):
    os.mkdir(output_dir_train)
if not os.path.exists(output_dir_test):
    os.mkdir(output_dir_test)
    
train_size = 60


for idx, img in enumerate(os.listdir(input_dir)):
    if img[0] == '.':
        continue
    new_img = cv2.imread(os.path.join(input_dir, img))[..., ::-1].astype(float) 
    new_img /= new_img.max()
    new_img = k_downscale(new_img, 3)
    if idx < train_size:
        plt.imsave(output_dir_train+img, new_img)
    else:
        plt.imsave(output_dir_test+img, new_img)
    
print("downscaled img size:", new_img.shape)
    
#######################################################
# Preprocessing
# 06/09/2019 -  Cropped and resized raw images
#
#######################################################

import os
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import cv2
import time
from tqdm import tqdm
from math import floor, ceil
import random
# import sys

# Define global variables
DEBUG_MODE = False
image_path = "./Images"
out_path = "./Resized"
img_ext = ['.jpg', '.jpeg', '.png']
RESIZED_HEIGHT = 224
RESIZED_WIDTH = 224
rand_seed = 22

class MyImage:
    img = None
    name = ''
    opt =  cv2.IMREAD_UNCHANGED

    def __init__(self, name, opt):
        self.name = name
        self.img = cv2.imread(name, opt)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def loadImages(path, subfolder = 'butterflies'):
    # Put files into lists and return them as one list of size 4
    image_files = sorted([os.path.join(path, subfolder, file)
         for file in os.listdir(path + "/" + subfolder) if file.lower().endswith(tuple(img_ext))])
    return image_files

def saveImages(myImgArr, path, randomise = True, subfolder = 'butterflies', test_train_validation_split = [0.2, 0.8*0.8, 0.8*0.2]):

    n = len(myImgArr)    
    sum_split = sum(test_train_validation_split)
    split = [s/sum_split for s in test_train_validation_split]

    # split according to index
    num_test = floor(split[0]*n)
    num_train = floor(split[1]*n)
    num_validation = n - num_test - num_train

    test_lb = 0
    test_ub = test_lb + num_test - 1
    train_lb = test_ub + 1
    train_ub = train_lb + num_train - 1
    validation_lb = train_ub + 1
    validation_ub = validation_lb + num_validation - 1

    bounds = {
        'test': [test_lb, test_ub],
        'train': [train_lb, train_ub],
        'validation': [validation_lb, validation_ub]
    }

    for p in ['test', 'train', 'validation']:
        fullFolderPath = os.path.join(path, p, subfolder)
        try:
            os.makedirs(fullFolderPath)
            print("  Directory " + fullFolderPath + " created")
        except FileExistsError:
            print("  Directory " + fullFolderPath + " already exists")
        
        print("  - Saving " + p + " dataset in " + fullFolderPath)
        cB = bounds[p]

        if (randomise):
            random.seed(rand_seed)
            random.shuffle(myImgArr, )

        for i in tqdm(range(cB[0], cB[1]+1)):
            mImage = myImgArr[i]
            baseName = os.path.basename(mImage.name)
            fullPath = os.path.join(fullFolderPath, baseName)
            if (DEBUG_MODE): print(fullPath)
            cv2.imwrite( fullPath, mImage.img )

# Display two images
def display(a, b, mainTitle = "", title1 = "Original", title2 = "Edited"):
    a_rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB) # convert from BGR to RGB
    b_rgb = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
    plt.subplot(121), plt.imshow(a_rgb), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b_rgb), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.suptitle(mainTitle)
    # plt.show()

def cropAndresize(data, subfolder='butterflies'):
    # Load image
    print("  Loading images into memory. This might take a while")
    mImg = [MyImage(i, cv2.IMREAD_UNCHANGED) for i in tqdm(data)]
    
    # --------------------------------
    # Set dim of the resize
    height = RESIZED_HEIGHT
    width = RESIZED_WIDTH
    dim = (width, height)
    res_img = []
    print("  Cropping and resizing images")
    for i in tqdm(range(len(mImg))):
        # Crop image to a square from the center
        ih = mImg[i].img.shape[0]
        iw = mImg[i].img.shape[1]
        # print(ih, iw)
        if (ih > iw):
            crop_img = mImg[i].img[int((ih-iw)/2):int((ih-iw)/2)+iw, 0:iw]
        else:
            crop_img = mImg[i].img[0:ih, int((iw-ih)/2):int((iw-ih)/2)+ih]

        # Resize        
        res = cv2.resize(crop_img, dim, interpolation=cv2.INTER_LINEAR)
        new = AttrDict()
        new.img = res
        new.name = mImg[i].name
        res_img.append(new)
        if (DEBUG_MODE): print('Original size',mImg[i].img.shape, '\nResized', res.shape)
    
    # Visualize one of the images in the array
    import random
    r = random.randint(0, len(mImg)-1)
    display(mImg[r].img, res_img[r].img, mainTitle = mImg[r].name, title1="Original " + str(mImg[r].img.shape), title2="Cropped+Resized " + str(res_img[r].img.shape))

    # Save to output folder
    saveImages(res_img, out_path, subfolder = subfolder)

def main():
    global image_path

    for i in ['butterflies', 'moths', 'bees']:
        print("------------------------------")
        print("Preprocessing " + i + " folder")
        print("------------------------------")
        start_time = time.time()
        print("  Loading image names.")
        dataset = loadImages(image_path, subfolder = i)

        # Send all the images to pre-processing
        # cropAndresize(dataset[:10], subfolder = i)
        cropAndresize(dataset, subfolder= i)
        elapsed_time = time.time() - start_time

        print("  Completed in " + str(elapsed_time))
        plt.show()
main()
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import random
from PIL import Image, ImageOps

# Dataset Directory
DATADIR = "G:\School\Shapes Detect\Gesture Model\Shapes"

#List of categories for tensorflow class labels
CATEGORIES = [ 
    "circle", 
    "square", 
    "triangle",
    "x", 
    "double-line-horizontal", 
    "double-line-vertical", 
    "triple-line-horizontal", 
    "triple-line-vertical",
    "diagonal-from-right",
    "diagonal-from-left"
]

# Image Data array
training_data = []

# Image size for standard (28 for optimal handwritten stuff)
IMG_SIZE = 28

def create_training_data():
    # Iterate through categories
    for category in CATEGORIES:
        # Image directory
        path = os.path.join(DATADIR, category)
        #Class number for tensorflow
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                # read image
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                #resize for standardization
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                #Raw image
                training_data.append([new_array, class_num])

                #manipulate and edit images to easily tripple image dataset

                #Flipped image
                flipped_array = cv2.flip(new_array, 0)
                training_data.append([flipped_array, class_num])
                #Mirrored image
                mirrored_array = cv2.flip(new_array, 1)
                training_data.append([mirrored_array, class_num])
                #Flipped mirror image
                flipped_mirrored_array = cv2.flip(new_array, -1)
                training_data.append([flipped_mirrored_array, class_num])

                
            except Exception as e:
                print(e)

create_training_data()

print(len(training_data))

#Shuffle training data
random.shuffle(training_data)

x = []
y = []
#append features and labels to x and y array
for features, label in training_data:
    x.append(features)
    y.append(label)

#convert arrays to numpy array so its readable later
x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

#pickle array
pickle_out = open("x.pickle", "wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


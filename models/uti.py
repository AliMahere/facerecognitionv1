import tensorflow as tf
import cv2
import numpy as np


tf_version = tf.__version__
tf_major_version = int(tf_version.split(".")[0])
tf_minor_version = int(tf_version.split(".")[1])

if tf_major_version == 1:
	import keras
	from keras.preprocessing.image import load_img, save_img, img_to_array
	from keras.applications.imagenet_utils import preprocess_input
	from keras.preprocessing import image
elif tf_major_version == 2:
	from tensorflow import keras
	from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
	from tensorflow.keras.applications.imagenet_utils import preprocess_input
	from tensorflow.keras.preprocessing import image


def find_input_shape(model):
    # https://github.com/serengil/deepface/blob/2ed4b226245937cbe2b354b54d728d4dc529201d/deepface/commons/functions.py#L237
    #face recognition models have different size of inputs
    #my environment returns (None, 224, 224, 3) but some people mentioned that they got [(None, 224, 224, 3)]. I think this is because of version issue.

    input_shape = model.layers[0].input_shape

    if type(input_shape) == list:
        input_shape = input_shape[0][1:3]
    else:
        input_shape = input_shape[1:3]

    #----------------------
    #issue 289: it seems that tf 2.5 expects you to resize images with (x, y)
    #whereas its older versions expect (y, x)

    if tf_major_version == 2 and tf_minor_version >= 5:
        x = input_shape[0]; y = input_shape[1]
        input_shape = (y, x)

    #----------------------

    if type(input_shape) == list: #issue 197: some people got array here instead of tuple
        input_shape = tuple(input_shape)

    return input_shape

def normalize(img, normalization = 'non'):
    if normalization == 'non':
        return img
    elif normalization == 'VGGFace':
        # mean subtraction based on VGGFace1 training data
        img[..., 0] -= 93.5940
        img[..., 1] -= 104.7624
        img[..., 2] -= 129.1863

    elif(normalization == 'VGGFace2'):
        # mean subtraction based on VGGFace2 training data
        img[..., 0] -= 91.4953
        img[..., 1] -= 103.8827
        img[..., 2] -= 131.0912
def prepareInput(img, target_size):
        	#---------------------------------------------------
#resize image to expected shape

# img = cv2.resize(img, target_size) #resize causes transformation on base image, adding black pixels to resize will not deform the base image
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.shape[0] > 0 and img.shape[1] > 0:
        factor_0 = target_size[0] / img.shape[0]
        factor_1 = target_size[1] / img.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
        img = cv2.resize(img, dsize)

        # Then pad the other side to the target size by adding black pixels
        diff_0 = target_size[0] - img.shape[0]
        diff_1 = target_size[1] - img.shape[1]

        img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')

    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)
        
    img_pixels = image.img_to_array(img) #what this line doing? must?
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels /= 255 #normalize input in [0, 1]
    img_pixels = normalize(img = img_pixels)
        
    return img_pixels

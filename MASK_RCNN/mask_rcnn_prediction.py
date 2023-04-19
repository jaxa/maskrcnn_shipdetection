"""
Mask R-CNN for Ship Detection
Ship detection model for benchmark
Copyright (c) 2023 Japan Aerospace Explration Agency.
All Rights Reserved.

This file is coverd by the LICENSE.txt in the root of this project.
"""

debug = True
# debug = False

import os
import random
import time

import mrcnn.model as modellib
import numpy as np
from mask_rcnn_model import DetectorConfig
from skimage.io import imread

import gc; gc.enable() # memory is tight


os.environ['CUDA_VISIBLE_DEVICES']="0"

DATA_DIR = './test_data/'
ROOT_DIR = './'

test_dicom_dir = os.path.join(DATA_DIR, 'test_v2')

MODEL_PATH = "./model/mask_rcnn_airbus_0018.h5"

class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def main():
    start = time.time()

    test_names = os.listdir(test_dicom_dir)
    if debug:
        test_names = test_names[:10]

    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode='inference', 
                            config=inference_config,
                            model_dir=ROOT_DIR)

    # Load trained weights (fill in path to trained weights here)
    model_path = MODEL_PATH
    assert model_path != "", "Provide path to trained weights"
    print("*"*5 + "Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    
    print("*"*5 + "Estimated number of")
    for i in range(len(test_names)):
        image_id = random.choice(test_names)  

        image = imread(os.path.join(test_dicom_dir, image_id))
        # resize_factor = 1 ## ORIG_SIZE / config.IMAGE_SHAPE[0]

        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1) 
                            
        results = model.detect([image]) #, verbose=1)
        r = results[0]
        print(f"{i+1}st: ", (r['rois']))

    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

if __name__ =="__main__":
    main()

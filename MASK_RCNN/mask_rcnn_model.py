# debug = True
debug = False

import os
import csv
import time
import numpy as np
import pandas as pd 

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib

from skimage.io import imread
from skimage.morphology import label
import gc; gc.enable() # memory is tight

from sklearn.model_selection import train_test_split
import warnings 


os.environ['CUDA_VISIBLE_DEVICES']="1"
DATA_DIR = '/data/SHIP_AIRBUS/'
ROOT_DIR = './'

train_dicom_dir = os.path.join(DATA_DIR, 'train_v2')
test_dicom_dir = os.path.join(DATA_DIR, 'test_v2')
COCO_WEIGHTS_PATH = "mask_rcnn_coco.h5"

# Some setup functions and classes for Mask-RCNN
class DetectorConfig(Config):    
    # Give the configuration a recognizable name  
    NAME = 'airbus'
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 9
    
    BACKBONE = 'resnet50'
    
    NUM_CLASSES = 2  # background and ship classes
    
    IMAGE_MIN_DIM = 384
    IMAGE_MAX_DIM = 384
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 64
    MAX_GT_INSTANCES = 14
    DETECTION_MAX_INSTANCES = 10
    DETECTION_MIN_CONFIDENCE = 0.95
    DETECTION_NMS_THRESHOLD = 0.0

    STEPS_PER_EPOCH = 15 if debug else 150
    VALIDATION_STEPS = 10 if debug else 125
    
    ## balance out losses
    LOSS_WEIGHTS = {
        "rpn_class_loss": 30.0,
        "rpn_bbox_loss": 0.8,
        "mrcnn_class_loss": 6.0,
        "mrcnn_bbox_loss": 1.0,
        "mrcnn_mask_loss": 1.2
    }

def multi_rle_encode(img, **kwargs):
    '''
    Encode connected regions as separated masks
    '''
    labels = label(img)
    if img.ndim > 2:
        return [rle_encode(np.sum(labels==k, axis=2), **kwargs) for k in np.unique(labels[labels>0])]
    else:
        return [rle_encode(labels==k, **kwargs) for k in np.unique(labels[labels>0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return '' ## no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return '' ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks

def masks_as_color(in_mask_list):
    # Take the individual ship masks and create a color mask array for each ships
    all_masks = np.zeros((768, 768), dtype = np.float)
    scale = lambda x: (len(in_mask_list)+x+1) / (len(in_mask_list)*2) ## scale the heatmap image to shift 
    for i,mask in enumerate(in_mask_list):
        if isinstance(mask, str):
            all_masks[:,:] += scale(i) * rle_decode(mask)
    return all_masks


class DetectorDataset(utils.Dataset):
    """Dataset class for training our dataset.
    """

    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__(self)
        
        # Add classes
        self.add_class('ship', 1, 'Ship')
        
        # add images 
        for i, fp in enumerate(image_fps):
            annotations = image_annotations.query('ImageId=="' + fp + '"')['EncodedPixels']
            self.add_image('ship', image_id=i, path=os.path.join(train_dicom_dir, fp), 
                           annotations=annotations, orig_height=orig_height, orig_width=orig_width)
            
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        image = imread(fp)
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                mask[:, :, i] = rle_decode(a)
                class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)


def main():
    start = time.time()
    
    config = DetectorConfig()
    config.display()

    # montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)

    exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                    '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg', 
                    'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                    'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg'] #corrupted images

    train_names = [f for f in os.listdir(train_dicom_dir) if f not in exclude_list]
    test_names = [f for f in os.listdir(test_dicom_dir) if f not in exclude_list]

    # training dataset
    SEGMENTATION = DATA_DIR + '/train_ship_segmentations_v2.csv'
    anns = pd.read_csv(SEGMENTATION)

    train_names = anns[anns.EncodedPixels.notnull()].ImageId.unique().tolist()  ## override with ships

    test_size = config.VALIDATION_STEPS * config.IMAGES_PER_GPU
    image_fps_train, image_fps_val = train_test_split(train_names, test_size=test_size, random_state=42)
    
    f = open('val_dataset.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(image_fps_val)
    f.close()

    if debug:
        image_fps_train = image_fps_train[:100]
        image_fps_val = image_fps_val[:100]
        test_names = test_names[:100]

    # Examine the annotation data, parse the dataset, and view dicom fields¶
    image_fps, image_annotations = train_names, anns
    ds = imread(os.path.join(train_dicom_dir, image_fps[0])) # read  image from filepath 
    ORIG_SIZE = ds.shape[0]

    # Create and prepare the training dataset using the DetectorDataset class.
    # prepare the training dataset
    print("*"*10+ "Create and prepare the training dataset using the DetectorDataset class. ")

    dataset_train = DetectorDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
    dataset_train.prepare()

    # prepare the validation dataset
    dataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
    dataset_val.prepare()


    # Now it's time to train the model. Note that training even a basic model can take a few hours.
    model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)

    # Exclude the last layers because they require a matching
    # number of classes
    model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])

    LEARNING_RATE = 0.003

    # Train Mask-RCNN Model 
    print("*"*10+ "Train Mask-RCNN Model ")
    warnings.filterwarnings("ignore")

    # train heads with higher lr to speedup the learning
    model.train(dataset_train, dataset_val,
                learning_rate=LEARNING_RATE/2,
                epochs=5 if debug else 20,
                layers='all',
                augmentation=None)  ## no need to augment yet

    history = model.keras_model.history.history
    
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

if __name__ =="__main__":
    main()
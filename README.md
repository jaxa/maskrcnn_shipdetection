# Mask R-CNN for Ship Detection
This repository is sample code for ship detection using [Mask R-CNN](https://arxiv.org/abs/1703.06870), implemented on Python 3.7, Keras, and TensorFlow.


## Build & RUN on Linux
```bash
# 1. Pre-req's
$ git clone xxx

# 2. Add model & test data
Add test data in ./MASK-RCNN/test_data/
Add the trained model in ./MASK-RCNN/model/

# 3. docker build
$ docker build -t mask-rcnn .

# 4. Run
$ docker run -it --rm --name mrc mask-rcnn
```

## Dependency
- Python
  -  [Python3.7](https://www.python.org/)

## License
This project is under the Apache 2.0 license. Please find LICENSE.txt for further information.

## Reference 
https://www.kaggle.com/code/hmendonca/airbus-mask-rcnn-and-coco-transfer-learning  
https://www.kaggle.com/code/hmendonca/airbus-mask-rcnn-and-coco-transfer-learning/data  
https://github.com/matterport/Mask_RCNN


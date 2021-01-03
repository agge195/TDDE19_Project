# TDDE19_Project# Instance Segmentation (Mask R-CNN) on Cigbuts dataset

# Getting Started

## Requirements
Python 2.8.1, TensorFlow 2.1, Keras 2.3 and  other common packages listed in <code>requirements.txt</code>

### akTwelve Requirements
<a href="https://github.com/akTwelve/Mask_RCNN">Aktwelve</a> has a forked version of MatterPort's original version of Mask R-CNN with some bug fixes and updated versions (using TensorFlow 2 instead of TensorFlow 1).

## Installation
### akTwelve
1. Clone https://github.com/akTwelve/Mask_RCNN.git
2. Change directory and install dependencies

    <code>
    cd Mask_RCNN<br>
    pip3 install -r requirements.txt
    </code>
3. Run the setup file

    <code>python3 setup.py install</code>
4. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the <a href="https://github.com/matterport/Mask_RCNN/releases">releases page</a>.
5. (Optional) To train or test on MS COCO install pycocotools from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

    *   Linux: https://github.com/waleedka/coco
    * Windows: https://github.com/philferriere/cocoapi. You must have the Visual C++ 2015 build tools on your path (see the repo for additional details) 

### Our work
1. Clone this repository
2. Change directory and install dependencies

    <code>
    cd Mask_RCNN<br>
    pip3 install -r requirements.txt
    </code>
3. 



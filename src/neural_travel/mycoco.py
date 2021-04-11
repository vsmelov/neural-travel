import os
import sys
from optparse import OptionParser

import coco
import skimage.io

import mrcnn.model as modellib
from mrcnn import utils
from neural_travel.vis_regions import vis_regions_to_file

# Root directory of the project
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print('REPO_DIR {}'.format(REPO_DIR))
ROOT_DIR = os.path.join(REPO_DIR, 'Mask_RCNN')

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version

# Directory to save logs and trained model
MODEL_DIR = os.path.join(REPO_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(REPO_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


def parse_args():
    parser = OptionParser()
    parser.add_option(
        "--img",
        dest="img",
        help="path to image",
        metavar="IMAGE",
    )
    parser.add_option(
        "--out",
        dest="out",
        help="out path",
        metavar="IMAGE",
        default='out.jpg',
    )
    options, _args = parser.parse_args()
    return options


args = parse_args()


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()



# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Load a random image from the images folder
image = skimage.io.imread(args.img)

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
vis_regions_to_file(
    image, r['rois'], r['masks'], r['class_ids'],
    class_names, r['scores'],
    out_image_path=args.out,
)


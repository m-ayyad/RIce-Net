#### STATIONS ####
STATIONS = "WI_Milwaukee_River_near_Cedarburg"

#### DATASET ####
# base path of the dataset
DATA_PATH = "data"
POLYGON = [[1920,418],[1439,378],[931,371],[720,367],[130,644],[226,738],[166,1080],[1920,1080]]
CROP = [312,None,128,None]
PIXELS = 614847.0

# mean and std
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# define the input image dimensions
INPUT_IMAGE_WIDTH = 1152
INPUT_IMAGE_HEIGHT = 640
NUM_CHANNELS = 3

# define the test split
TEST_BATCH_SIZE = 1

#### CLASSES ####
# Define classes with the same order as the mask
CLASSES = ['Open Water', 'Ice Presence']

#### DEVICES ####
# determine the device to be used for training and evaluation
USE_CUDA = True

#### MODEL ####
CLASSIFICATION_MODEL = "weights/classification_model.pth"
SEGMENTATION_MODEL = "weights/segmentation_model.pth"

#### OUTPUT ####
# show figure
SHOW = True


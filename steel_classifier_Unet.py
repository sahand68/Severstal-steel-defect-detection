import fastai
from fastai.vision import *
from fastai.callbacks import SaveModelCallback
import gc
import os
from sklearn.model_selection import KFold
from PIL import Image
import zipfile
import io
import cv2

sz = 256
bs = 16
nfolds = 4
SEED = 2019
TRAIN = '../input/severstal-256x256-images-with-defects/images/'
MASKS = '../input/severstal-256x256-images-with-defects/masks/'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    #tf.set_random_seed(seed)
seed_everything(SEED)
torch.backends.cudnn.benchmark = True
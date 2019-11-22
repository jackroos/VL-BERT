import sys
from PIL import Image

import warnings

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

try:
    im = Image.open(sys.argv[1]).convert('RGB')
    # remove images with too small or too large size
    if (im.size[0] < 10 or im.size[1] < 10 or im.size[0] > 10000 or im.size[1] > 10000):
        raise Exception('')
except:
    print(sys.argv[1])

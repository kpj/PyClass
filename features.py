"""
Define all features to be extracted from the data
"""

from PIL import Image
from PIL.ImageStat import Stat

from skimage.feature import local_binary_pattern


class BaseFeatureExtractor(object):
    """ Basis for all feature extractors
    """
    def extract(self, data):
        raise NotImplementedError('No way of extracting features specified')

class BasicImageStats(BaseFeatureExtractor):
    """ Compute some basic pixel-based image statistics
    """
    def extract(self, img_path):
        stats = Stat(Image.open(img_path))
        return stats.count \
            + stats.sum \
            + stats.sum2 \
            + stats.mean \
            + stats.median \
            + stats.rms \
            + stats.var \
            + stats.stddev

class LocalBinaryPatterns(BaseFeatureExtractor):
    """ Extract some LBPs
    """
    def extract(self, img_path):
        image = Image.open(img_path)

        box = (100, 100, 500, 500)
        sub_img = image.crop(box)

        lbp = local_binary_pattern(sub_img.getdata(), 8 * 3, 3, 'uniform')
        return lbp.flat

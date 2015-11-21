"""
Define all features to be extracted from the data
"""

from PIL.ImageStat import Stat


class BaseFeatureExtractor(object):
    """ Basis for all feature extractors
    """
    def extract(self, data):
        raise NotImplementedError('No way of extracting features specified')

class BasicImageStats(BaseFeatureExtractor):
    """ Compute some basic pixel-based image statistics
    """
    def extract(self, img):
        stats = Stat(img)
        return stats.count \
            + stats.sum \
            + stats.sum2 \
            + stats.mean \
            + stats.median \
            + stats.rms \
            + stats.var \
            + stats.stddev

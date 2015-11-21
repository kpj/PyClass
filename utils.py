"""
Functions of varying functionality
"""

import os


def get_feature_extractors():
    """ Return list of feature extractors given in `features.py`
    """
    import features
    return [getattr(features, x) for x in dir(features) if isinstance(getattr(features, x), type) and x != 'BaseFeatureExtractor' and getattr(features, x).__bases__[0] == features.BaseFeatureExtractor]

def get_feature_vector(img_path):
    """ Generate feature vector for given image
    """
    vec = []
    for Cls in get_feature_extractors():
        cur = Cls().extract(img_path)
        vec.extend(cur)

    return vec

def read_images(root_dir='./images'):
    """ Recursively read images in given root directory.
        Assume that each subdirectory specifies the class of the respective images
    """
    data = []
    for img_cls, class_dir in enumerate(os.listdir(root_dir)):
        for img_name in os.listdir(os.path.join(root_dir, class_dir)):
            data.append((
                class_dir, os.path.join(root_dir, class_dir, img_name)
            ))
    return data

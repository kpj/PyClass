"""
Classify images using random decision forests
"""

import sys

from mlearner import train_model
from utils import get_feature_vector


def classify_image(image_path, clf):
    """ Classify given image
    """
    fvecs = [get_feature_vector(image_path)]
    print(clf.predict(fvecs))

def main():
    if len(sys.argv) != 2:
        print('Usage: %s <image>' % sys.argv[0])
        sys.exit(1)

    clf = train_model()
    classify_image(sys.argv[1], clf)


if __name__ == '__main__':
    main()

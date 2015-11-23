"""
Classify images using random decision forests
"""

import sys, os

from mlearner import train_model
from utils import get_feature_vector


def classify_image(image_path, clf):
    """ Classify given image
    """
    fvecs = [get_feature_vector(image_path)]
    print(clf.predict(fvecs), clf.predict_proba(fvecs))

def compute_score(root_dir, clf):
    """ Test model with all images in given directory
    """
    fvecs = []
    true_classes = []

    for fn in os.listdir(root_dir):
        assert len(fn.split('_')) > 1, 'Invalid filename'
        fname = os.path.join(root_dir, fn)

        fvecs.append(get_feature_vector(fname))
        true_classes.append(fn.split('_')[0])

    print('Accuracy:', clf.score(fvecs, true_classes))

def main():
    """ Execute action depending on external input
    """
    if len(sys.argv) != 2:
        print('Usage: %s <image|directory>' % sys.argv[0])
        sys.exit(1)

    clf = train_model()

    arg = sys.argv[1]
    if os.path.isfile(arg):
        classify_image(arg, clf)
    elif os.path.isdir(arg):
        compute_score(arg, clf)
    else:
        print('Invalid argument...')


if __name__ == '__main__':
    main()

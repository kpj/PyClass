"""
Train ML algorithm
"""

import os

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from utils import read_images, get_feature_vector


def create_forest():
    """ Create new random decision forest
    """
    images = read_images()

    # assign classes
    image_classes = [icls for icls, _ in images]

    # extract features
    feature_vectors = []
    for icls, img_path in images:
        cur = get_feature_vector(img_path)
        feature_vectors.append(cur)

    assert len(feature_vectors) > 0, 'Must have at least one feature vector to train on'

    # good estimator number for classification tasks
    sqrt_feat_num = int(np.sqrt(len(feature_vectors[0])))

    print('Creating random forest:')
    print(' ', sqrt_feat_num, 'estimator%s' % ('' if sqrt_feat_num == 1 else 's'))

    # create forest
    clf = RandomForestClassifier(n_estimators=sqrt_feat_num, n_jobs=-1)
    clf = clf.fit(feature_vectors, image_classes)

    return clf

def train_model(cache_file='cache_dir/model.pkl'):
    """ Train model or use cached version if available
    """
    if os.path.isfile(cache_file):
        print('Loading cached model "%s"' % cache_file)
        clf = joblib.load(cache_file)
    else:
        clf = create_forest()
        joblib.dump(clf, cache_file)

    return clf

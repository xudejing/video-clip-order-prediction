"""Retrieve video samples."""
import os
import math
import itertools
import argparse
import time
import random
import json

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

from datasets.ucf101 import UCF101ClipRetrievalDataset
from datasets.hmdb51 import HMDB51ClipRetrievalDataset
from models.c3d import C3D
from models.r3d import R3DNet
from models.r21d import R2Plus1DNet


def topk_retrieval():
    """Extract features from test split and search on train split features."""
    print('Load local .npy files.')
    X_train = np.load('data/features/ucf101/r3d/train_feature.npy')
    y_train = np.load('data/features/ucf101/r3d/train_class.npy')
    X_train = np.mean(X_train,1)
    y_train = y_train[:,0]
    X_train = X_train.reshape((-1, X_train.shape[-1]))
    y_train = y_train.reshape(-1)

    X_test = np.load('data/features/ucf101/r3d/test_feature.npy')
    y_test = np.load('data/features/ucf101/r3d/test_class.npy')
    X_test = np.mean(X_test,1)
    y_test = y_test[:,0]
    X_test = X_test.reshape((-1, X_test.shape[-1]))
    y_test = y_test.reshape(-1)

    distances = cosine_distances(X_test, X_train)
    indices = np.argsort(distances)

    class_idx2label = pd.read_csv('data/ucf101/split/classInd.txt', header=None, sep=' ').set_index(0)[1]

    top_k_indices = indices[:, :2]
    test_id = 0
    for ind, class_idx in zip(top_k_indices, y_test):
        idxs = y_train[ind]
        print('{}, [{}]-{}, {}'.format(test_id, class_idx2label[class_idx], [ class_idx2label[i] for i in idxs ], ind))
        test_id += 1


def topk_retrieval_cross_dataset():
    """Extract features from test split and search on train split features."""
    print('Load local .npy files.')
    X_train = np.load('data/features/hmdb51/r3d/train_feature.npy')
    y_train = np.load('data/features/hmdb51/r3d/train_class.npy')
    X_train = np.mean(X_train,1)
    y_train = y_train[:,0]
    X_train = X_train.reshape((-1, X_train.shape[-1]))
    y_train = y_train.reshape(-1)

    X_test = np.load('data/features/ucf101/r3d/test_feature.npy')
    y_test = np.load('data/features/ucf101/r3d/test_class.npy')
    X_test = np.mean(X_test,1)
    y_test = y_test[:,0]
    X_test = X_test.reshape((-1, X_test.shape[-1]))
    y_test = y_test.reshape(-1)

    distances = cosine_distances(X_test, X_train)
    indices = np.argsort(distances)

    ucf101_class_idx2label = pd.read_csv('data/ucf101/split/classInd.txt', header=None, sep=' ').set_index(0)[1]
    hmdb51_class_idx2label = pd.read_csv('data/hmdb51/split/classInd.txt', header=None, sep=' ').set_index(0)[1]

    top_k_indices = indices[:, :3]
    test_id = 0
    for ind, class_idx in zip(top_k_indices, y_test):
        idxs = y_train[ind]
        print('{}, [{}]-{}, {}'.format(test_id, ucf101_class_idx2label[class_idx], [ hmdb51_class_idx2label[i] for i in idxs ], ind))
        test_id += 1


if __name__ == '__main__':
    topk_retrieval()
    # topk_retrieval_cross_dataset()
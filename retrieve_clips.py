"""Video retrieval experiment, top-k."""
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


def load_pretrained_weights(ckpt_path):
    """load pretrained weights and adjust params name."""
    adjusted_weights = {}
    pretrained_weights = torch.load(ckpt_path)
    for name, params in pretrained_weights.items():
        if 'base_network' in name:
            name = name[name.find('.')+1:]
            adjusted_weights[name] = params
            print('Pretrained weight name: [{}]'.format(name))
    return adjusted_weights


def extract_feature(args):
    """Extract and save features for train split, several clips per video."""
    torch.backends.cudnn.benchmark = True
    # Force the pytorch to create context on the specific device 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    ########### model ##############
    if args.model == 'c3d':
        model = C3D(with_classifier=False, return_conv=True).to(device)
    elif args.model == 'r3d':
        model = R3DNet(layer_sizes=(1,1,1,1), with_classifier=False, return_conv=True).to(device)
    elif args.model == 'r21d':
        model = R2Plus1DNet(layer_sizes=(1,1,1,1), with_classifier=False, return_conv=True).to(device)

    if args.ckpt:
        pretrained_weights = load_pretrained_weights(args.ckpt)
        model.load_state_dict(pretrained_weights, strict=True)
    model.eval()
    torch.set_grad_enabled(False)
    ### Exract for train split ###
    train_transforms = transforms.Compose([
        transforms.Resize((128, 171)),
        transforms.CenterCrop(112),
        transforms.ToTensor()
    ])
    if args.dataset == 'ucf101':
        train_dataset = UCF101ClipRetrievalDataset('data/ucf101', 16, 10, True, train_transforms)
    elif args.dataset == 'hmdb51':
        train_dataset = HMDB51ClipRetrievalDataset('data/hmdb51', 16, 10, True, train_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False,
                                    num_workers=args.workers, pin_memory=True, drop_last=True)
    
    features = []
    classes = []
    for data in tqdm(train_dataloader):
        sampled_clips, idxs = data
        clips = sampled_clips.reshape((-1, 3, 16, 112, 112))
        inputs = clips.to(device)
        # forward
        outputs = model(inputs)
        # print(outputs.shape)
        # exit()
        features.append(outputs.cpu().numpy().tolist())
        classes.append(idxs.cpu().numpy().tolist())

    features = np.array(features).reshape(-1, 10, outputs.shape[1])
    classes = np.array(classes).reshape(-1, 10)
    np.save(os.path.join(args.feature_dir, 'train_feature.npy'), features)
    np.save(os.path.join(args.feature_dir, 'train_class.npy'), classes)

    ### Exract for test split ###
    test_transforms = transforms.Compose([
        transforms.Resize((128, 171)),
        transforms.CenterCrop(112),
        transforms.ToTensor()
    ])
    if args.dataset == 'ucf101':
        test_dataset = UCF101ClipRetrievalDataset('data/ucf101', 16, 10, False, test_transforms)
    elif args.dataset == 'hmdb51':
        test_dataset = HMDB51ClipRetrievalDataset('data/hmdb51', 16, 10, False, test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                    num_workers=args.workers, pin_memory=True, drop_last=True)

    features = []
    classes = []
    for data in tqdm(test_dataloader):
        sampled_clips, idxs = data
        clips = sampled_clips.reshape((-1, 3, 16, 112, 112))
        inputs = clips.to(device)
        # forward
        outputs = model(inputs)
        features.append(outputs.cpu().numpy().tolist())
        classes.append(idxs.cpu().numpy().tolist())

    features = np.array(features).reshape(-1, 10, outputs.shape[1])
    classes = np.array(classes).reshape(-1, 10)
    np.save(os.path.join(args.feature_dir, 'test_feature.npy'), features)
    np.save(os.path.join(args.feature_dir, 'test_class.npy'), classes)


def topk_retrieval(args):
    """Extract features from test split and search on train split features."""
    print('Load local .npy files.')
    X_train = np.load(os.path.join(args.feature_dir, 'train_feature.npy'))
    y_train = np.load(os.path.join(args.feature_dir, 'train_class.npy'))
    X_train = np.mean(X_train,1)
    y_train = y_train[:,0]
    X_train = X_train.reshape((-1, X_train.shape[-1]))
    y_train = y_train.reshape(-1)

    X_test = np.load(os.path.join(args.feature_dir, 'test_feature.npy'))
    y_test = np.load(os.path.join(args.feature_dir, 'test_class.npy'))
    X_test = np.mean(X_test,1)
    y_test = y_test[:,0]
    X_test = X_test.reshape((-1, X_test.shape[-1]))
    y_test = y_test.reshape(-1)

    ks = [1, 5, 10, 20, 50]
    topk_correct = {k:0 for k in ks}

    distances = cosine_distances(X_test, X_train)
    indices = np.argsort(distances)

    for k in ks:
        # print(k)
        top_k_indices = indices[:, :k]
        # print(top_k_indices.shape, y_test.shape)
        for ind, test_label in zip(top_k_indices, y_test):
            labels = y_train[ind]
            if test_label in labels:
                # print(test_label, labels)
                topk_correct[k] += 1

    for k in ks:
        correct = topk_correct[k]
        total = len(X_test)
        print('Top-{}, correct = {:.2f}, total = {}, acc = {:.3f}'.format(k, correct, total, correct/total))

    with open(os.path.join(args.feature_dir, 'topk_correct.json'), 'w') as fp:
        json.dump(topk_correct, fp)


def parse_args():
    parser = argparse.ArgumentParser(description='Frame Retrieval Experiment')
    parser.add_argument('--cl', type=int, default=16, help='clip length')
    parser.add_argument('--model', type=str, default='c3d', help='c3d/r3d/r21d')
    parser.add_argument('--dataset', type=str, default='ucf101', help='ucf101/hmdb51')
    parser.add_argument('--feature_dir', type=str, default='data/features/ucf101/c3d', help='dir to store feature.npy')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--ckpt', type=str, help='checkpoint path')
    parser.add_argument('--bs', type=int, default=8, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(vars(args))

    if not os.path.exists(args.feature_dir):
        os.makedirs(args.feature_dir)
        extract_feature(args)
    topk_retrieval(args)

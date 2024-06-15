import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import torch

#
class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.dataset in ['NELL','WiKi']:
            self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.feature_dim = self.train_feature.shape[1]
        self.att_dim = self.attribute.shape[1]
        self.text_dim = self.att_dim
        self.tr_cls_centroid = np.zeros([self.seenclasses.shape[0], self.feature_dim], np.float32)

        for i in range(self.seenclasses.shape[0]):
            self.tr_cls_centroid[i] = np.mean(self.train_feature[self.train_label == i].numpy(), axis=0)

    def read_matdataset(self, opt):
        ######################################
        # if opt.dataset == "Wiki":
        #     opt.dataset = "WiKi" # k->K
        #     print(True)
        # label = np.load(opt.data_presave_path + '/'+ opt.dataset + '/entity_pair_label' + opt.extractor_way +'.npz')['LABEL']
        label = np.load(opt.data_presave_path + '/' + opt.dataset + '/entity_pair_label_addloss' + '.npz')['LABEL']
        # ([  0,   0,   0, ..., 112, 112, 112])
        seen_label = np.unique(label)   # 获取数组 label 中的唯一值，即所有出现过的标签。
        feature = np.load(opt.data_presave_path + '/'+ opt.dataset + '/entity_pair_matrix_addloss' + '.npz')['EPM']
        # feature = np.load(opt.data_presave_path + '/' + opt.dataset + '/entity_pair_matrix' + opt.extractor_way + '.npz')['EPM']
        # [[ 1.6808336   0.45335892    ...    0.2787776  0.33077055 ]
        #  ...
        #  [ 1.6528339   0.5654188     ...   0.85813725  -0.13600463]]
        # number : [[181053][200]]
        attribute = np.load(opt.data_presave_path +'/'+  opt.dataset + r'/rela_matrix_.npz')['relaM']     # 实体之间关系的属性信息


        self.attribute = attribute
        if opt.dataset == 'NELL':
            self.attribute = self.attribute/10
        # print(self.attribute)

        _train_feature = feature
        self.train_feature = torch.from_numpy(_train_feature).float()
        self.train_label = torch.from_numpy(label).long()
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
            # [1,1,1,2,2,3,3,4,4,4] --> [1,2,3,4]
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
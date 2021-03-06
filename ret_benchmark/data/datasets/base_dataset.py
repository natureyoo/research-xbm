# encoding: utf-8

# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import os, re, json
from collections import defaultdict
import numpy as np

from torch.utils.data import Dataset
from ret_benchmark.utils.img_reader import read_image


def read_data(dataset_name='DeepFashion2', bbox_gt=True, type_list=['train', 'validation']):
    base_path = os.path.join('/home/jayeon', dataset_name)

    img_list = {}
    item_dict = {}

    if dataset_name == 'DeepFashion':
        file_info = {}
        for idx, line in enumerate(open(os.path.join(base_path, 'Anno/list_bbox_inshop.txt'), 'r').readlines()):
            if idx > 1:     # except first 2 lines
                file_info[line.strip().split()[0]] = np.asarray(line.strip().split()[1:], dtype=np.int)

        # build category idx dictionary
        category_set = set([cate for gender in ['WOMEN', 'MEN'] for cate in os.listdir(os.path.join(base_path, 'img', gender))])
        category_dict = {cate: idx for idx, cate in enumerate(category_set)}

        item_dict = {item.strip(): idx - 1 for idx, item in
                     enumerate(open(os.path.join(base_path, 'Anno/list_item_inshop.txt'), 'r').readlines()) if idx > 0}

        for file_type in type_list:
            img_list[file_type] = []
            is_train = file_type == 'train'
            for idx, line in enumerate(open(os.path.join(base_path, 'Eval/list_eval_partition.txt'), 'r').readlines()):
                if idx > 1 and is_train == (line.strip().split()[2] == 'train'):        # except first 2 lines
                    file_name = line.strip().split()[0]
                    img_list[file_type].append(
                        [file_name, item_dict[file_name.split('/')[-2]], category_dict[file_name.split('/')[2]],
                         file_info[file_name][2:]])

            img_list[file_type] = np.asarray(img_list[file_type], dtype=object)

    elif dataset_name == 'DeepFashion2':
        box_key = 'bounding_box' if bbox_gt else 'proposal_boxes'
        for file_type in type_list:
            anno_dir_path = os.path.join(base_path, file_type, 'annos') if bbox_gt \
                else os.path.join('/home/jayeon/TmpData', file_type, 'annos_f')
            img_list[file_type] = []
            item_dict[file_type] = {}
            item_idx = 0
            for file_name in os.listdir(os.path.join(base_path, file_type, 'image')):
                anno_path = os.path.join(anno_dir_path, file_name.split('.')[0] + '.json')
                if not os.path.exists(anno_path):
                    continue
                anno = json.load(open(anno_path, 'r'))
                source_type = 0 if anno['source'] == 'user' else 1
                pair_id = str(anno['pair_id'])
                for key in anno.keys():
                    if key not in ['source', 'pair_id'] and int(anno[key]['style']) > 0:
                        bounding_box = np.asarray(anno[key][box_key], dtype=int)
                        cate_id = anno[key]['category_id'] - 1
                        pair_style = '_'.join([pair_id, str(anno[key]['style'])])
                        if pair_style not in item_dict[file_type].keys():
                            item_dict[file_type][pair_style] = item_idx
                            item_idx += 1
                        img_list[file_type].append([os.path.join(file_type, 'image', file_name),
                                                    item_dict[file_type][pair_style], cate_id,
                                                    bounding_box, source_type])

            img_list[file_type] = np.asarray(img_list[file_type], dtype=object)

    return img_list, base_path, item_dict


class BaseDataSet(Dataset):
    """
    Basic Dataset read image path from img_source
    img_source: list of img_path and label
    """

    def __init__(self, img_source, is_train, transforms=None, mode="RGB", unsup_da=False):
        self.mode = mode
        self.transforms = transforms
        self.root = os.path.dirname(img_source)
        assert os.path.exists(img_source), f"{img_source} NOT found."
        self.img_source = img_source
        self.is_train = is_train
        self.unsup_da = unsup_da

        self.label_list = list()
        self.path_list = list()
        self._load_data()
        self.label_index_dict = self._build_label_index_dict()

    def __len__(self):
        return len(self.label_list)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"| Dataset Info |datasize: {self.__len__()}|num_labels: {len(set(self.label_list))}|"

    def _load_data(self):
        if self.unsup_da:
            source_type = 'train' if self.is_train else 'validation'
            img_list, base_path, item_dict = read_data(dataset_name='DeepFashion2', bbox_gt=True)
            labeled_data = [path.split('/')[-1].split('.')[0] for path
                            in img_list[source_type][img_list[source_type][:, 4] == 1][:, 0]]
            with open(self.img_source, "r") as f:
                for line in f:
                    _path, _label = re.split(r",", line.strip())
                    if _path.split('/')[-1].split('_')[0] in labeled_data:
                        self.path_list.append(_path)
                        self.label_list.append(_label)

        else:
            with open(self.img_source, "r") as f:
                for line in f:
                    _path, _label = re.split(r",", line.strip())
                    self.path_list.append(_path)
                    self.label_list.append(_label)

    def _build_label_index_dict(self):
        index_dict = defaultdict(list)
        for i, label in enumerate(self.label_list):
            index_dict[label].append(i)
        return index_dict

    def __getitem__(self, index):
        path = self.path_list[index]
        img_path = os.path.join(self.root, path)
        label = self.label_list[index]

        img = read_image(img_path, mode=self.mode)

        if self.transforms is not None:
            img = self.transforms(img)
        return img, label, index

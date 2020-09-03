import sys, os, json
import numpy as np


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


def get_distance_mtx(source_mtx, target_mtx):
    dist = - 2 * source_mtx.dot(target_mtx.transpose()) + (source_mtx ** 2).sum(axis=1).reshape(-1, 1) + (target_mtx ** 2).sum(axis=1).reshape(1, -1)
    return dist

img_list, base_path, item_dict = read_data(dataset_name='DeepFashion2', bbox_gt=True, type_list=['train', 'validation'])
img_list['validation'][:, 0] = np.asarray([f.split('/')[-1].split('.')[0] for f in img_list['validation'][:, 0]])
result = np.load('./output/2020-08-19_feat.npz')
validation_set = np.asarray([(line.split(',')[0], line.split(',')[1].strip()) for line in open('/home/jayeon/DeepFashion2/val_img_cropped_label.txt', 'r').readlines()])
source = np.zeros(len(result['upc']), dtype=int)
for idx, (feat, label, file_name) in enumerate(zip(result['feat'], result['upc'], validation_set[:, 0])):
    f = file_name.split('/')[-1].split('_')[0]
    source[idx] = img_list['validation'][:, 4][img_list['validation'][:, 0] == f][0]

emb_mtx = result['feat']
label = result['upc']

user_idx = np.where(source == 0)[0]
shop_idx = np.where(source == 1)[0]
user_emb_mtx = emb_mtx[user_idx]
shop_emb_mtx = emb_mtx[shop_idx]

user_shop_dist = get_distance_mtx(user_emb_mtx, shop_emb_mtx)
user_shop_rank = np.argsort(user_shop_dist, axis=1)[:, :100]
user_shop_corr = np.zeros_like(user_shop_rank)
for idx, (u_idx, rel_shop) in enumerate(zip(user_idx, user_shop_rank)):
    u_label = label[u_idx]
    user_shop_corr[idx] = label[shop_idx[rel_shop]] == u_label

for k in [1, 5, 10, 20, 100]:
    topk_accuracy = np.sum(np.sum(user_shop_corr[:, :k], axis=1) > 0) / user_shop_corr.shape[0]
    print('Top-{} Accuracy: {:.5f}'.format(k, topk_accuracy))

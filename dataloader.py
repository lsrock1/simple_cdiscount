import copy
import numpy as np
import random, multiprocessing
from collections import defaultdict
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler


from PIL import Image
import os.path as osp
import os
import bson
from tqdm import tqdm
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
from transforms import build_transforms


def read_cat_id(path='dataset'):
    csv_file = osp.join(path, 'category_names.csv')
    cat_ids = {}
    with open(csv_file, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            cat_id, cat_name_1, cat_name_2, cat_name_3 = row
            cat_ids[int(cat_id)] = [cat_name_1, cat_name_2, cat_name_3]
    return cat_ids


def process_dirs(data_source, pids_to_classid, catid_to_classid):
    
    def func(pid):
        imgs = glob(osp.join(pid, '*.jpg'))
        if len(imgs) <= 3:
            return
        # origin_pid = pid
        pid = osp.basename(pid)
        pid, catid = [int(i) for i in pid.split('_')]
        # catid, pid = [int(i) for i in pid.split('_')]
        pids_to_classid[pid] = len(pids_to_classid)
        catid_to_classid[catid] = len(catid_to_classid)

        for img in imgs:
            queue.append([img, pids_to_classid[pid], catid_to_classid[catid]])

    return func


def read_accepted(path='dataset'):
    r = []
    with open(os.path.join(path, 're_categories.txt'), 'r') as f:
        for l in f.readlines():
            if len(l.strip()) > 0:
                r.append(int(l))
    return r


def read_ssg(path='ssg'):
    a_images = sorted(glob(os.path.join(path, 'SSGDF_MAIN_IMAGE_', '*/*')))
    b_images = sorted(glob(os.path.join(path, 'search/*/*')))
    pids_to_classid = {}
    catid_to_classid = {}

    for name in tqdm(a_images + b_images):
        ori = name
        name = os.path.basename(name)
        splitted_name = name.split('_')
        catid = splitted_name[0]
        pid = '_'.join(splitted_name[1:])
        
        if pid not in pids_to_classid:
            pids_to_classid[pid] = len(pids_to_classid)

        if catid not in catid_to_classid:
            catid_to_classid[catid] = len(catid_to_classid)

        data_source.append([ori, pids_to_classid[pid], catid_to_classid[catid]])
    return data_source, catid_to_classid


def read_data(path='dataset', is_test=False, max_pids=50000):
    accepted_catids = read_accepted()
    data_source = []
    path = osp.join(path, 'test' if is_test else 'train_filtered')
    pids_to_classid = {}
    catid_to_classid = {}
    pids = sorted(glob(osp.join(path, '*')))
    print('start..')
    # pool = multiprocessing.Pool(processes=12)
    # process_func = process_dirs(data_source, pids_to_classid, catid_to_classid)

    # with tqdm(total=len(pids)) as t:
    #     for _ in pool.imap_unordered(process_func, pids):
    #         t.update(1)
    # if is_test:
    #     with open('catid_to_classid.pickle', 'rb') as handle:
    #         catid_to_classid = pickle.load(handle)

    for pid in tqdm(pids[:max_pids]):
        imgs = glob(osp.join(pid, '*.jpg'))
        if len(imgs) <= 3:
            continue
        # origin_pid = pid
        pid = osp.basename(pid)
        pid, catid = [int(i) for i in pid.split('_')]
        if catid not in accepted_catids:
            continue
        # catid, pid = [int(i) for i in pid.split('_')]
        pids_to_classid[pid] = len(pids_to_classid)
        if catid not in catid_to_classid:
            catid_to_classid[catid] = len(catid_to_classid)

        for img in imgs:
            data_source.append([img, pids_to_classid[pid], catid_to_classid[catid]])
    print('total the number of id: ', len(pids_to_classid))
    print('total the number of cat id: ', len(catid_to_classid))
    print('total the number of data: ', len(data_source))
    return data_source, catid_to_classid


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError('{} does not exist'.format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print('IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # print(index)
        # print(self.dataset)
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path


class RandomIdentitySampler(Sampler):
    """Randomly samples N identities each with K instances.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        batch_size (int): batch size.
        num_instances (int): number of instances per identity in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        if batch_size < num_instances:
            raise ValueError(
                'batch_size={} must be no less '
                'than num_instances={}'.format(batch_size, num_instances)
            )

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        # TODO: improve precision
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        print('sampling..')
        for pid in tqdm(self.pids):
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(
                    idxs, size=self.num_instances, replace=True
                )
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
        flatten_batch_idxs = []
        flatten_batch_idxs_pid = []

        for pid, idxs in batch_idxs_dict.items():
            for idx in idxs:
                flatten_batch_idxs_pid.append(pid)
                flatten_batch_idxs.append(idx)
        # flatten_batch_idxs_pid = np.array(flatten_batch_idxs_pid).squeeze()
        flatten_batch_idxs = np.array(flatten_batch_idxs).squeeze()
        # print(flatten_batch_idxs.shape)
        shuffle_idx = np.arange(flatten_batch_idxs.shape[0])
        np.random.shuffle(shuffle_idx)
        # print(shuffle_idx)

        # flatten_batch_idxs_pid = flatten_batch_idxs_pid[shuffle_idx]
        flatten_batch_idxs = flatten_batch_idxs[shuffle_idx].reshape(-1)
        # print(flatten_batch_idxs_pid.shape)
        # print(flatten_batch_idxs.shape)
        return iter(flatten_batch_idxs.tolist())
        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


def build_dataloader(is_test, batch_size):
    data_source, catid_to_classid = read_ssg() #read_data(is_test=is_test, max_pids=-1)
    sampler = RandomIdentitySampler(data_source, batch_size, 4)
    transforms, test_transforms = build_transforms(128,
                     128,
                     illumination_aug=False,
                     random_erase=False,  # use random erasing for data augmentation
                     color_jitter=True,  # randomly change the brightness, contrast and saturation
                     color_aug=False)
    transforms = test_transforms if is_test else transforms
    return DataLoader(
            ImageDataset(data_source, transform=transforms),# sampler=sampler,
            batch_size=batch_size, shuffle=True, num_workers=6,
            pin_memory=True, drop_last=True
        ), catid_to_classid

from model import build_model
from transforms import build_transforms
from dataloader import build_dataloader, read_image
import torch
import time
import os
import cv2
from annoy import AnnoyIndex
from glob import glob
from tqdm import tqdm
import numpy as np


def distance(vectors):
    m = vectors.size(0)
    vectors_pow = torch.pow(vectors, 2).sum(dim=1, keepdim=True).expand(m, m)
    distmat = vectors_pow + vectors_pow.t()
    # distmat = torch.pow(vectors, 2).sum(dim=1, keepdim=True).expand(m, m) + \
    #     torch.pow(test_features, 2).sum(dim=1, keepdim=True).expand(m, m).t()
    distmat.addmm_(1, -2, vectors, vectors.t())
    distmat = distmat.clamp(min=1e-12).sqrt()

    return distmat.cpu().numpy()


def main():
    load = '0610_model59_10000pids.pth'#'new_model4.pth'
    # if len(load) > 0:
    load_model = torch.load(load)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # dataloader, _ = build_dataloader(is_test=True, batch_size=128)
    # with open('catid_to_classid.pickle', 'rb') as handle:
    #     catid_to_classid = pickle.load(handle)
    model = build_model(1414)
    model = torch.nn.DataParallel(model).cuda()
    # criterion = TripletLoss(margin=0.3)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.00005)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
    
    model.load_state_dict(load_model['model'])
    # optimizer.load_state_dict(load_model['optimizer'])
    
    results = []

    data = get_data()
    _, transforms = build_transforms(128,
                     128,
                     illumination_aug=False,
                     random_erase=False,  # use random erasing for data augmentation
                     color_jitter=True,  # randomly change the brightness, contrast and saturation
                     color_aug=False)
    # num_epochs = 60
    model = model.cuda()
    with torch.no_grad():
        for d in tqdm(data):
            img = read_image(d)
            img = transforms(img)
            vector = model(img.unsqueeze(0).cuda())
            results.append(vector) # .cpu().squeeze(0).numpy()
        test = transforms(read_image('t.jpg'))
        # results = [model(test.unsqueeze(0))] + results
    results = torch.cat(results, dim=0)
    distmat = distance(results)
        # results = [model(test.unsqueeze(0)).cpu().squeeze(0).numpy()] + results
    # f = 2048
    # t = AnnoyIndex(f, 'euclidean')  # Length of item vector that will be indexed
    # for idx, v in enumerate(results):
    #     # v = [random.gauss(0, 1) for z in range(f)]
    #     t.add_item(idx, v)

    # t.build(100) # 10 trees
    # t.save('test.ann')
    # t = AnnoyIndex(f, 'euclidean')
    # t.load('test.ann')
    # test = read_image('t.jpeg')
    # vector
    print(distmat.shape)
    for i, dist in enumerate(distmat):
        val = []
        if i != 0:
            val.append(read_image(data[i]))
        indices = np.argsort(dist)[:5]
        # print(dist[indices])
        for idx in indices:
            img = read_image(data[idx])
            val.append(img)
        img = np.concatenate(val, axis=1).astype(np.uint8)
        cv2.imshow('tt', img)
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            break
    # for img_idx in range(len(data)):
    #     val = []
    #     for tmp in t.get_nns_by_item(img_idx, 5):
    #         img = read_image(data[tmp])
    #         val.append(img)
    #     img = np.concatenate(val, axis=1).astype(np.uint8)
    #     cv2.imshow('tt', img)
    #     cv2.waitKey(0)
    # ...

    # u = AnnoyIndex(f, 'angular')
    # u.load('test.ann') # super fast, will just mmap the file
    # print(u.get_nns_by_item(0, 1000))


def get_data():
    # in train 10000 pids
    dirs = sorted(glob('dataset/train/*'))[10000:12000]
    imgs_total = []
    for d in tqdm(dirs):
        imgs = glob(os.path.join(d, '*.jpg'))
        if len(imgs) <= 3:
            continue
        imgs_total += imgs

    return imgs_total


if __name__ == '__main__':
    main()
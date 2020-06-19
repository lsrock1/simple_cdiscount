from model import build_model
from dataloader import build_dataloader, read_cat_id
from pytorch_metric_learning import losses, miners
from loss import TripletLoss
import torch.nn.functional as F
import torch
import time
import os
import pickle


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        if isinstance(output, (tuple, list)):
            output = output[0]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(100.0 / batch_size)
            res.append(acc.item())
        return res


class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    load = ''#'new_model4.pth'
    if len(load) > 0:
        load_model = torch.load(load)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    dataloader, catid_to_classid = build_dataloader(is_test=False, batch_size=64)
    # with open('catid_to_classid.pickle', 'wb') as handle:
    #     pickle.dump(catid_to_classid, handle, protocol=pickle.HIGHEST_PROTOCOL)
    num_classes = len(catid_to_classid)
    model = build_model(num_classes).cuda()
    # model = torch.nn.DataParallel(model).cuda()
    criterion = TripletLoss(margin=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.00005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
    
    if len(load) > 0:
        model.load_state_dict(load_model['model'])
        optimizer.load_state_dict(load_model['optimizer'])
    
    num_epochs = 60
    for epoch in range(num_epochs):
        
        run_epoch(model, criterion, dataloader, epoch, scheduler, optimizer)
        scheduler.step()
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, f'ssg_new_model{epoch}.pth')


def run_epoch(model, criterion, dataloader, epoch, scheduler, optimizer):
    print_freq = 10
    htri_losses = AverageMeter()
    xent_losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    accs = AverageMeter()

    end = time.time()
    for batch_idx, (imgs, pids, camid, _) in enumerate(dataloader):
        data_time.update(time.time() - end)
        imgs, pids, camid = imgs.cuda(), pids.cuda(), camid.cuda()

        outputs, clas = model(imgs)
        # print(pids)
        htri_loss = criterion(outputs, pids)
        # clas = model[1(outputs)
        # xent_loss = F.cross_entropy(clas, camid)
        loss = htri_loss# + xent_loss
        # loss = xent_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        htri_losses.update(htri_loss.item(), pids.size(0))
        # xent_losses.update(xent_loss.item(), pids.size(0))
        accs.update(accuracy(clas, camid)[0])
        if (batch_idx + 1) % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Xent {xent.val:.4f} ({xent.avg:.4f})\t'
                  'Htri {htri.val:.4f} ({htri.avg:.4f})\t'
                  'Acc {acc.val:.2f} ({acc.avg:.2f})\t'.format(
                epoch + 1, batch_idx + 1, len(dataloader),
                batch_time=batch_time,
                data_time=data_time,
                xent=xent_losses,
                htri=htri_losses,
                acc=accs
            ))

        end = time.time()



if __name__ == '__main__':
    main()
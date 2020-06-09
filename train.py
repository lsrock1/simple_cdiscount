from model import build_model
from dataloader import build_dataloader
from pytorch_metric_learning import losses
import torch
import time


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
    dataloader = build_dataloader(is_test=False, batch_size=64)
    model = build_model().cuda()
    criterion = losses.TripletMarginLoss(margin=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    num_epochs = 100
    for epoch in range(num_epochs):
        
        run_epoch(model, criterion, dataloader, epoch, scheduler, optimizer)
        scheduler.step()
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, f'new_model{epoch}.pth')


def run_epoch(model, criterion, dataloader, epoch, scheduler, optimizer):
    print_freq = 10
    htri_losses = AverageMeter()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    for batch_idx, (imgs, pids, camid, _) in enumerate(dataloader):
        data_time.update(time.time() - end)
        imgs, pids = imgs.cuda(), pids.cuda()
        outputs = model(imgs)
        # print(pids)
        htri_loss = criterion(outputs, pids)

        loss = htri_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        htri_losses.update(htri_loss.item(), pids.size(0))

        if (batch_idx + 1) % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Htri {htri.val:.4f} ({htri.avg:.4f})\t'.format(
                epoch + 1, batch_idx + 1, len(dataloader),
                batch_time=batch_time,
                data_time=data_time,
                htri=htri_losses,
            ))

        end = time.time()



if __name__ == '__main__':
    main()
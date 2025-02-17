import time
import torch
import numpy as np
import torch.nn.parallel
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import os
from utilis.meters import AverageMeter
from utilis.meters import ProgressMeter
from utilis.matrix import accuracy
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def precision_recall_f1score(y_true, y_pred, num_classes):
    # predicted = torch.max(y_pred, dim=1)[1]
    predicted = torch.argmax(torch.softmax(y_pred,dim=1), dim=1)
    confusion_matrix = torch.zeros(num_classes, num_classes)
    for t, p in zip(y_true, predicted):
        confusion_matrix[t, p] += 1

    precision = confusion_matrix.diag() / confusion_matrix.sum(dim=0)
    recall = confusion_matrix.diag() / confusion_matrix.sum(dim=1)
    f1_score = 2 * precision * recall / (precision + recall + 1e-7)

    precision[torch.isnan(precision)] = 0
    recall[torch.isnan(recall)] = 0
    f1_score[torch.isnan(f1_score)] = 0
    precision = torch.mean(precision)
    recall = torch.mean(recall)
    f1_score = torch.mean(f1_score)
    return precision, recall, f1_score


def validate(val_loader, model, criterion, num_classes, test=True, args=None):
    if test:
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        precision = AverageMeter('Val Precision', ':6.2f')
        recall = AverageMeter('Val Recall', ':6.2f')
        f1_score = AverageMeter('Val f1_score', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix='Test: ')
    else:
        batch_time = AverageMeter('val Time', ':6.3f')
        losses = AverageMeter('val Loss', ':.4e')
        top1 = AverageMeter('Val Acc@1', ':6.2f')
        top5 = AverageMeter('Val Acc@5', ':6.2f')
        precision = AverageMeter('Val Precision', ':6.2f')
        recall = AverageMeter('Val Recall', ':6.2f')
        f1_score = AverageMeter('Val f1_score', ':6.2f')
        
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix='Val: ')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            images = images.to(device)
            target = target.to(device)


            output= model(images)
            loss = criterion(output, target)
            pre, re, f1 = precision_recall_f1score(target, output, num_classes)
            acc1 = accuracy(output, target, topk=(1, ))[0]
            # print("acc1:",acc1[0])
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            # top5.update(acc5[0], images.size(0))
            precision.update(pre, images.size(0))
            recall.update(re, images.size(0))
            f1_score.update(f1, images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                method_name = args.log_path.split('/')[-2]
                progress.display(i, method_name)
                progress.write_log(i, args.log_path)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        with open(args.log_path, 'a') as f1:
            f1.writelines(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                          .format(top1=top1, top5=top5))


    return top1.avg, precision, recall, f1_score

import argparse
import os
import time
import sys

import torch
import torch.optim as optim
import torchvision.transforms as transforms

from misc.dataset import CocoCaptionsRV, Shopping
from misc.evaluation import eval_recall
from misc.loss import HardNegativeContrastiveLoss
from misc.model import joint_embedding
from misc.utils import AverageMeter, save_checkpoint, collate_fn_padded, log_epoch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import multiprocessing


def train(train_loader, model, criterion, optimizer, epoch, print_freq=1000):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model = model.train()

    end = time.time()
    for i, (imgs, caps, lengths) in enumerate(train_loader):
        if i%2 == 1:
                print("%2.2f"% (i/len(train_loader)*100), '\%', end='\r')
        input_imgs, input_caps = imgs.cuda(), caps.cuda()

        data_time.update(time.time() - end)
        
        optimizer.zero_grad()
        end = time.time()
        output_imgs, output_caps = model(input_imgs, input_caps, lengths)
        loss = criterion(output_imgs, output_caps)
        
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), imgs.size(0))

        batch_time.update(time.time() - end)
        
        if i % print_freq == 0 or i == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses))
        end = time.time()

    return losses.avg, batch_time.avg, data_time.avg
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-pf", dest="print_frequency", help="Number of element processed between print", type=int, default=10)
    parser.add_argument("-bs", "--batch_size", help="The size of the batches", type=int, default=400)
    parser.add_argument("-lr", "--learning_rate", dest="lr", help="Initialization of the learning rate", type=float, default=0.001)
    parser.add_argument("-lrd", "--learning_rate_decrease", dest="lrd",
                        help="List of epoch where the learning rate is decreased (multiplied by first arg of lrd)", nargs='+', type=float, default=[0.5, 2, 3, 4, 5, 6])
    parser.add_argument("-fepoch", dest="fepoch", help="Epoch start finetuning resnet", type=int, default=8)
    parser.add_argument("-mepoch", dest="max_epoch", help="Max epoch", type=int, default=60)
    parser.add_argument('-sru', dest="sru", type=int, default=4)
    parser.add_argument("-de", dest="dimemb", help="Dimension of the joint embedding", type=int, default=2400)
    parser.add_argument("-d", dest="dataset", help="Dataset to choose : coco or shopping", default='coco')
    parser.add_argument("-dict", dest='dict', help='Dictionnary link', default="./data/wiki.fr.bin")
    parser.add_argument("-es", dest="embed_size", help="Embedding size", default=300, type=int)
    parser.add_argument("-w", dest="workers", help="Nb workers", default=multiprocessing.cpu_count(), type=int)
    parser.add_argument("-df", dest="dataset_file", help="File with dataset", default="")
    parser.add_argument("-r", dest="resume", help="Resume training")
    
    fout = open("logTime", "w")
    
    args = parser.parse_args()
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    prepro = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    prepro_val = transforms.Compose([
        transforms.Resize((350, 350)),
        transforms.ToTensor(),
        normalize,
    ])
    criterion = HardNegativeContrastiveLoss().cuda()
    print("Initializing network...")
    
    join_emb = joint_embedding(args).cuda()
    
    
    for param in join_emb.cap_emb.parameters():
        param.requires_grad = False
    
    for param in join_emb.img_emb.parameters():
        param.requires_grad = False
        
    optimizer=optim.Adam(filter(lambda p: p.requires_grad, join_emb.parameters()), lr=0.001)
    
    coco_data_train = CocoCaptionsRV(args, sset="val", transform=prepro)
    
    train_loader = DataLoader(coco_data_train, batch_size=args.batch_size, shuffle=False, drop_last=True,
                              num_workers=args.workers, collate_fn=collate_fn_padded, pin_memory=True)
    
    eTime = time.time()                
    train_loss, batch_train, data_train = train(train_loader, join_emb, criterion, optimizer, 1, print_freq=args.print_frequency)
    endTime = time.time()
    
    fout.write(str(endTime - eTime) +'\t'+str(batch_train))
    print("Time per batch : ", batch_train)
    print("Epoch in :", endTime - eTime)
    
    for param in join_emb.cap_emb.parameters():
        param.requires_grad = True
    optimizer.add_param_group({'params': join_emb.cap_emb.parameters(), 'lr': optimizer.param_groups[0]
                                       ['lr'], 'initial_lr': args.lr})

    eTime = time.time() 
    train_loss, batch_train, data_train = train(train_loader, join_emb, criterion, optimizer, 1, print_freq=args.print_frequency)
    endTime = time.time()
    print("Time per batch : ", batch_train)
    print("Epoch in :", endTime - eTime)
    fout.write(str(endTime - eTime) +'\t'+str(batch_train))
    
    print("Adding layers of img_emb")
    
    for i in range(len(join_emb.img_emb.module.base_layer)):
        for param in join_emb.img_emb.module.base_layer[i].parameters():
            param.requires_grad = True
        optimizer.add_param_group({'params': join_emb.img_emb.module.base_layer[i].parameters()
                                       , 'lr': optimizer.param_groups[0]
                                       ['lr'], 'initial_lr': args.lr})
        eTime = time.time() 
        train_loss, batch_train, data_train = train(train_loader, join_emb, criterion, optimizer, 1, print_freq=args.print_frequency)
        endTime = time.time()
        print("Time per batch : ", batch_train)
        print("Epoch in :", endTime - eTime)
        fout.write(str(endTime - eTime) +'\t'+str(batch_train))
    
    

    
    
    

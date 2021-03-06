import argparse
import os
import time
import sys

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np

from dataset.dataset import CocoCaptionsRV, Shopping, Multi30k, DoubleDataset
from utils.evaluation import eval_recall, k_recall
from utils.loss import HardNegativeContrastiveLoss
from models.model import joint_embedding
from utils.utils import AverageMeter, save_checkpoint, collate_fn_padded, log_epoch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import multiprocessing



def train(train_loader, model, criterion, optimizer, epoch, print_freq=1000):
    #amp_handle = amp.init()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model = model.train()

    end = time.time()
    print("Start Training")
    for i, (imgs, caps, lengths) in enumerate(train_loader):
        print("%2.2f"% (i/len(train_loader)*100), '\%', end='\r')
        
        input_imgs, input_caps = imgs.cuda(), caps.cuda()
        
        if (input_imgs != input_imgs).any():
            print("NaN found in input_imgs")
            sys.exit(0)
        if (input_caps != input_caps).any():
            print("NaN found in input_caps")
            sys.exit(0)
        

        data_time.update(time.time() - end)

        optimizer.zero_grad()
        output_imgs, output_caps = model(input_imgs, input_caps, lengths)
        
        if (output_imgs != output_imgs).any():
            print("NaN found in output image")
            sys.exit(0)
        if (output_caps != output_caps).any():
            print("NaN found in output caption")
            sys.exit(0)
        
        
        loss = criterion(output_imgs, output_caps)
        
        if (loss != loss).any():
            print(input_caps[-1])
            print(output_caps[-1])
            print(loss[1])
        
        
        #with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
        #    scaled_loss.backward()
        loss.backward()
        optimizer.step()
        
        

        losses.update(loss.item(), imgs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 or i == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses))

    return losses.avg, batch_time.avg, data_time.avg


def validate(val_loader, model, criterion, print_freq=1000):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model = model.eval()

    imgs_enc = list()
    caps_enc = list()
    end = time.time()
    for i, (imgs, caps, lengths) in enumerate(val_loader):

        input_imgs, input_caps = imgs.cuda(), caps.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        with torch.no_grad():
            output_imgs, output_caps = model(input_imgs, input_caps, lengths)
            loss = criterion(output_imgs, output_caps)

        imgs_enc.append(output_imgs.cpu().data.numpy())
        caps_enc.append(output_caps.cpu().data.numpy())
        losses.update(loss.item(), imgs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 or i == (len(val_loader) - 1):
            print('Data: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      i, len(val_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses))

    recall  = eval_recall(imgs_enc, caps_enc)
    print(recall)
    return losses.avg, batch_time.avg, data_time.avg, recall


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument("-n", '--name', required=True, help='Name of the model')
    parser.add_argument("-pf", dest="print_frequency", help="Number of element processed between print", type=int, default=10)
    parser.add_argument("-bs", "--batch_size", help="The size of the batches", type=int, default=400)
    parser.add_argument("-lr", "--learning_rate", dest="lr", help="Initialization of the learning rate", type=float, default=0.001)
    parser.add_argument("-lrd", "--learning_rate_decrease", dest="lrd",
                        help="List of epoch where the learning rate is decreased (multiplied by first arg of lrd)", nargs='+', type=float, default=[0.5, 2, 3, 4, 5, 6])
    parser.add_argument("-fepoch", dest="fepoch", help="Epoch start finetuning resnet", type=int, default=8)
    parser.add_argument("-mepoch", dest="max_epoch", help="Max epoch", type=int, default=60)
    parser.add_argument('-sru', dest="sru", type=int, default=4)
    parser.add_argument("-de", dest="dimemb", help="Dimension of the joint embedding", type=int, default=2400)
    parser.add_argument("-d", dest="dataset", help="Dataset to choose : coco, shopping, multi30k or double", default='coco')
    parser.add_argument("-dict", dest='dict', help='Dictionnary link', default="./data/wiki.fr.bin")
    parser.add_argument("-es", dest="embed_size", help="Embedding size", default=300, type=int)
    parser.add_argument("-w", dest="workers", help="Nb workers", default=multiprocessing.cpu_count(), type=int)
    parser.add_argument("-df", dest="dataset_file", help="File with dataset", default="")
    parser.add_argument("-r", dest="resume", help="Resume training")
    parser.add_argument("-pt", dest="pretrained", help="Path to pretrained model", default="False")
    parser.add_argument("-la", dest="lang", help="Language used for the dataset", default="en")
    parser.add_argument("--embed_type", default="multi", help="multi, align, bivec, subword")
    

    args = parser.parse_args()

    logger = SummaryWriter(os.path.join("./logs/", args.name))
    
    
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
    
    
    
    end = time.time()
    print("Initializing network ...", end=" ")

    if args.resume: # Resume previous learning
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        join_emb = joint_embedding(checkpoint['args_dict'])
        join_emb.load_state_dict(checkpoint["state_dict"])
        join_emb = torch.nn.DataParallel(join_emb.cuda())
        
        last_epoch = checkpoint["epoch"]
        opti = checkpoint["optimizer"]
        print("Load from epoch :", last_epoch)
        
        last_epoch += 1   
            
        lr_scheduler = MultiStepLR(opti, args.lrd[1:], gamma=args.lrd[0])
        lr_scheduler.step(last_epoch)
        best_rec = checkpoint['best_rec']
        
    else:
        # Create new model
        if args.pretrained != "False": #load a pre-trained model
            checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
            join_emb = joint_embedding(checkpoint['args_dict'])
            join_emb.load_state_dict(checkpoint["state_dict"])
        else:
            join_emb = joint_embedding(args)
            
        join_emb = torch.nn.DataParallel(join_emb.cuda())
        
        # Froze text side of the model
        for param in join_emb.module.cap_emb.parameters():
            param.requires_grad = False
        
        
        opti = optim.Adam(filter(lambda p: p.requires_grad, join_emb.parameters()), lr=args.lr)
        #opti = apex.optimizers.FusedAdam(filter(lambda p: p.requires_grad, join_emb.parameters()), lr=args.lr)
        #opti = apex.fp16_utils.FP16_Optimizer(optimizer)
        lr_scheduler = MultiStepLR(opti, args.lrd[1:], gamma=args.lrd[0])
        last_epoch = 0
        best_rec = 0
        
    criterion = HardNegativeContrastiveLoss().cuda()
    
    

    print("Done in: " + str(time.time() - end) + "s")

    end = time.time()
    print("Loading Data ...", end=" ")

    if args.dataset == 'coco':
        print("Using coco dataset")
        coco_data_train = CocoCaptionsRV(sset="trainrv", transform=prepro, embed_type=args.embed_type, embed_size=args.embed_size)
        coco_data_val = CocoCaptionsRV(sset="val", transform=prepro_val, embed_type=args.embed_type, embed_size=args.embed_size)
    elif args.dataset == 'shopping':
        print("Using shopping dataset")
        if args.dataset_file == '':
            coco_data_train = Shopping('/data/shopping/', 'data/shoppingShort.txt', sset="trainrv", transform=prepro, embed_type=args.embed_type, embed_size=args.embed_size)
            coco_data_val = Shopping('/data/shopping/', 'data/shoppingShort.txt',sset="val", transform=prepro_val, embed_type=args.embed_type, embed_size=args.embed_size)
        else:
            coco_data_train = Shopping('/data/shopping/', args.dataset_file, sset="trainrv", transform=prepro)
            coco_data_val = Shopping('/data/shopping/', args.dataset_file,sset="val", transform=prepro_val)
    elif args.dataset == "multi30k":
        print("multi30k dataset in ", args.lang)
        coco_data_train = Multi30k(sset="train", lang=args.lang, transform=prepro, embed_type=args.embed_type)
        coco_data_val = Multi30k(sset="val", lang=args.lang, transform=prepro_val, embed_type=args.embed_type)
        
    elif args.dataset == "double":
        print("Double dataset, coco + multi30k")
        print("multi30k dataset in ", args.lang)
        d1_train = CocoCaptionsRV(sset="trainrv", transform=prepro, embed_type=args.embed_type, embed_size=args.embed_size)
        d2_train = Multi30k(sset="train", lang=args.lang, transform=prepro, embed_type=args.embed_type)
        d1_val = CocoCaptionsRV(sset="val", transform=prepro_val, embed_type=args.embed_type, embed_size=args.embed_size)
        d2_val = Multi30k(sset="val", lang=args.lang, transform=prepro_val, embed_type=args.embed_type)
        
        coco_data_train = DoubleDataset(d1_train, d2_train)
        coco_data_val = DoubleDataset(d1_val, d2_val)
        

    train_loader = DataLoader(coco_data_train, batch_size=args.batch_size*3, shuffle=True, drop_last=False,
                              num_workers=args.workers, collate_fn=collate_fn_padded, pin_memory=True)
    val_loader = DataLoader(coco_data_val, batch_size=args.batch_size*3, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_fn_padded, pin_memory=True, drop_last=False)
    print("Done in: " + str(time.time() - end) + "s")

    #print("Validation")
    #val_loss, batch_val, data_val, recall = validate(val_loader, join_emb, criterion, print_freq=args.print_frequency)
    
    # For each epoch
    for epoch in range(last_epoch, args.max_epoch):
        is_best = False
        print("Train")
        train_loss, batch_train, data_train = train(train_loader, join_emb, criterion, opti, epoch, print_freq=args.print_frequency)
        print("Validation")
        val_loss, batch_val, data_val, recall = validate(val_loader, join_emb, criterion, print_freq=args.print_frequency)


        #Check if is best model
        if(sum(recall[0]) + sum(recall[1]) > best_rec):
            best_rec = sum(recall[0]) + sum(recall[1])
            is_best = True
                
        
        state = {
            'epoch': epoch,
            'state_dict': join_emb.module.state_dict(),
            'best_rec': best_rec,
            'args_dict': args,
            'optimizer': opti,
        }

        # save state
        log_epoch(logger, epoch, train_loss, val_loss, opti.param_groups[0]
                  ['lr'], batch_train, batch_val, data_train, data_val, recall)
        save_checkpoint(state, is_best, args.name, epoch)

        # Optimizing the text pipeline after one epoch
        if epoch == 1:
            train_loader = DataLoader(coco_data_train, batch_size=args.batch_size*2, shuffle=True, drop_last=False,
                                  num_workers=args.workers, collate_fn=collate_fn_padded, pin_memory=True)
            val_loader = DataLoader(coco_data_val, batch_size=args.batch_size*2, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_fn_padded, pin_memory=True, drop_last=False)
                            
            for param in join_emb.module.cap_emb.parameters():
                param.requires_grad = True
            opti.add_param_group({'params': filter(lambda p: p.requires_grad, join_emb.module.cap_emb.parameters()), 'lr': opti.param_groups[0]
                                       ['lr'], 'initial_lr': args.lr})
            lr_scheduler = MultiStepLR(opti, args.lrd[1:], gamma=args.lrd[0])

        # Starting the finetuning of the whole model
        if epoch == args.fepoch:
            print("Sarting finetuning")
            
            train_loader = DataLoader(coco_data_train, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.workers, collate_fn=collate_fn_padded, pin_memory=True)
            val_loader = DataLoader(coco_data_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_fn_padded, pin_memory=True, drop_last=False)
            
            finetune = True
            for param in join_emb.parameters():
                param.requires_grad = True

            # Keep the first layer of resnet frozen
            for i in range(0, 6):
                for param in join_emb.module.img_emb.base_layer[0][i].parameters():
                    param.requires_grad = False

            opti.add_param_group({'params': filter(lambda p: p.requires_grad, join_emb.module.img_emb.base_layer.parameters()), 'lr': opti.param_groups[0]
                                       ['lr'], 'initial_lr': args.lr})
            lr_scheduler = MultiStepLR(opti, args.lrd[1:], gamma=args.lrd[0])
        print('lr_schedule')
        lr_scheduler.step(epoch)

    print('Finished Training')

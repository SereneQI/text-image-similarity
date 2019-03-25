import argparse
import time

import torch
import torchvision.transforms as transforms

from dataset.dataset import CocoCaptionsRV, Multi30k
from utils.evaluation import eval_recall, eval_recall5
from models.model import joint_embedding
from utils.utils import collate_fn_padded
from torch.utils.data import DataLoader


device = torch.device("cuda")
# device = torch.device("cpu") # uncomment to run with cpu

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate the model on cross modal retrieval task')
    parser.add_argument("-p", '--path', dest="model_path", help='Path to the weights of the model to evaluate', required=True)
    parser.add_argument("-bs", "--batch_size", help="The size of the batches", type=int, default=64)
    parser.add_argument('-tr', dest="dset", help="Using training dataset instead of validation", default="val")
    parser.add_argument('-ds', "--dataset", default="mutli30k", help='Choose between coco, multi30k or shopping')
    parser.add_argument('-d', '--dict', default="data/wiki.multi.en.vec")
    parser.add_argument("-la", "--lang", default="en")
    parser.add_argument("--wildcat", default=None)
    parser.add_argument("--eval_type", default=1, help="1 or 5", type=int)
    parser.add_argument("--embed_type", default="multi")

    args = parser.parse_args()

    print("Loading model from:", args.model_path)
    checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)

    join_emb = joint_embedding(checkpoint['args_dict']).cuda()
    join_emb.load_state_dict(checkpoint["state_dict"])
    #join_emb = torch.nn.DataParallel(join_emb.cuda())

    for param in join_emb.parameters():
        param.requires_grad = False

    join_emb = torch.nn.DataParallel(join_emb.cuda().eval())
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    prepro_val = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        normalize,
    ])

    if args.dataset == "coco":
        dataset = CocoCaptionsRV(args, sset=args.dset, transform=prepro_val)
    else:
        dataset = Multi30k(sset=args.dset, transform=prepro_val, lang=args.lang, embed_type=args.embed_type)

    print("Dataset size: ", len(dataset))

    dataset_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=6, collate_fn=collate_fn_padded, pin_memory=True)

    imgs_enc = list()
    caps_enc = list()

    print("### Beginning of evaluation ###")
    end = time.time()
    for i, (imgs, caps, lengths) in enumerate(dataset_loader, 0):
        input_imgs, input_caps = imgs.to(device), caps.to(device)

        with torch.no_grad():
            output_imgs, output_caps = join_emb(input_imgs, input_caps, lengths)

        imgs_enc.append(output_imgs.cpu().data.numpy())
        caps_enc.append(output_caps.cpu().data.numpy())

        if i % 100 == 99:
            print(str((i + 1) * args.batch_size) + "/" + str(len(dataset)) + " pairs encoded - Time per batch: " + str((time.time() - end)) + "s")

        end = time.time()
    
    if args.eval_type == 1:
        print(args.model_path, args.dset, eval_recall(imgs_enc, caps_enc))
    elif args.eval_type == 5:
        print(args.model_path, args.dset, eval_recall5(imgs_enc, caps_enc))
    else:
        print("Unknown evaluation type")

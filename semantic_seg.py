import argparse
import time

import numpy as np
import torch
import torchvision.transforms as transforms

from dataset.dataset import CocoSemantic
from utils.localization import compute_semantic_seg
from models.model import joint_embedding
from utils.utils import collate_fn_semseg
from torch.utils.data import DataLoader



device = torch.device("cuda")
# device = torch.device("cpu") # uncomment to run with cpu

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate model on semantic segmentation')
    parser.add_argument("-p", '--path', dest="model_path", help='Path to the weight of the model to evaluate')
    parser.add_argument("-bs", "--batch_size", help="The size of the batches", type=int, default=100)
    parser.add_argument("-ct", "--ctresh", help="Thresholding coeeficient to binarize heat maps", type=float, default=0.45)

    args = parser.parse_args()

    print("Loading model from:", args.model_path)
    checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)

    join_emb = joint_embedding(checkpoint['args_dict'])
    join_emb.load_state_dict(checkpoint["state_dict"])

    for param in join_emb.parameters():
        param.requires_grad = False

    join_emb = torch.nn.DataParallel(join_emb.cuda().eval())

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    in_dim = (400.0, 400.0)
    prepro_val = transforms.Compose([
        transforms.Resize((int(in_dim[0]), int(in_dim[1]))),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = CocoSemantic('/data/datasets/coco/', '/data/m.portaz/wiki.multi.en.vec' , transform=prepro_val)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_fn_semseg, pin_memory=True)

    imgs_enc = list()
    imgs_wld = list()
    target_ann = list()
    sizes_list = list()

    print("### Starting image embedding ###")
    end = time.time()
    for i, (imgs, sizes, targets) in enumerate(loader):

        input_imgs = imgs.cuda()

        with torch.no_grad():
            _, output_imgs = join_emb.module.img_emb.get_activation_map(input_imgs)

        output_imgs.size()
        target_ann += targets
        sizes_list += sizes
        imgs_enc.append(output_imgs.cpu().data.numpy())

        if i % 100 == 99:
            print(str((i + 1) * args.batch_size) + "/" + str(len(dataset)) + " images encoded - Time per batch: " + str((time.time() - end)) + "s")

        end = time.time()

    cats_enc = list()
    # process captions
    print("### Starting category embedding ###")
    for i, caps in enumerate(dataset.categories_w2v):

        input_caps = caps.unsqueeze(0).cuda()
        print("Input caps shape :", input_caps.shape)
        print(dataset.categories_lengths[i])
        with torch.no_grad():
            #_, output_caps = join_emb(None, input_caps, dataset.categories_lengths[i])
            _, output_caps = join_emb(None, input_caps, None)

        cats_enc.append(output_caps.squeeze().cpu().data.numpy())

    cats_stack = dict(zip([cat['name'] for cat in dataset.categories], cats_enc))

    imgs_stack = np.vstack(imgs_enc)
    #print(imgs_stack)
    print("Dimension of images maps:", imgs_stack.shape)
    print("Dimension of categories embeddings:", len(cats_enc))

    fc_w = join_emb.module.fc.weight.cpu().data.numpy()

    mAp_at_IoU = compute_semantic_seg(imgs_stack, sizes_list, target_ann, cats_stack, fc_w, args.ctresh)
    print("Coco semantic segmentation IoU@(0.3,0.4,0.5):", mAp_at_IoU)

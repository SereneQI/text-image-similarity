import json
import os
import re
import io

import numpy as np
import torch
import torch.utils.data as data

from misc.config import path
from misc.utils import encode_sentence, _load_dictionary, encode_sentence_fasttext, fr_preprocess, collate_fn_padded
from misc.model import joint_embedding
#from config import path
#from utils import encode_sentence, _load_dictionary, encode_sentence_fasttext, fr_preprocess, collate_fn_padded
from PIL import Image
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from torchvision import transforms
import argparse
import multiprocessing
#from visual_genome import local as vg

import fastText
from torch.utils.data import DataLoader




def _load_vec(emb_path):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id






class CocoCaptionsRV(data.Dataset):

    def __init__(self, args, root=path["COCO_ROOT"], coco_json_file_path=path["COCO_RESTVAL_SPLIT"], sset="train", transform=None):
        self.root = os.path.join(root, "images/")
        self.transform = transform

        # dataset.json come from Karpathy neural talk repository and contain the restval split of coco
        with open(coco_json_file_path, 'r') as f:
            datas = json.load(f)

        if sset == "train":
            self.content = [x for x in datas["images"] if x["split"] == "train"]
        elif sset == "trainrv":
            self.content = [x for x in datas["images"] if x["split"] == "train" or x["split"] == "restval"]
        elif sset == "val":
            self.content = [x for x in datas["images"] if x["split"] == "val"]
        else:
            self.content = [x for x in datas["images"] if x["split"] == "test"]

        self.content = [(os.path.join(y["filepath"], y["filename"]), [x["raw"] for x in y["sentences"]]) for y in self.content]

        #path_params = os.path.join(word_dict_path, 'utable.npy')
        #self.params = np.load(path_params, encoding='latin1')
        if args.dict[-3:] == 'vec':
            print("Using .vec file")
            self.embed, self.id2word, self.word2id = _load_vec(args.dict)
        else:
            print("Using .bin file")
            self.embed = fastText.load_model(args.dict)
            self.word2id = None
        #self.dico = _load_dictionary(word_dict_path)

    def __getitem__(self, index, raw=False):
        idx = index / 5

        idx_cap = index % 5

        path = self.content[int(idx)][0]
        target = self.content[int(idx)][1][idx_cap]

        if raw:
            return path, target

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        #target = encode_sentence(target, self.params, self.dico)
        if self.word2id is None:
            target = encode_sentence_fasttext(target, self.embed)
        else:
            target = encode_sentence(target, self.embed, self.word2id)
        return img, target

    def __len__(self):
        return len(self.content) * 5



class Shopping(data.Dataset):

    def __init__(self, args, root_dir, captionFile, transform, sset="train"):
        self.transform = transform

        self.imList = []
        self.capList = []

        f = open(captionFile)
        for i, line in enumerate(f):
            line = line.rstrip()
            im, cap = line.split('\t')
            if 1 <= len(cap.split(' ')) <= 20:
                self.imList.append(os.path.join(root_dir, im+'.jpg'))
                self.capList.append(cap.split(' '))
                    
        separation = len(self.imList)-(len(self.imList)//20)
        if sset == "train":
            self.imList = self.imList[:separation]
            self.capList = self.capList[:separation]
        elif sset == "val": #5 last % used for validation
            self.imList = self.imList[separation:]
            self.capList = self.capList[separation:]

        #path_params = os.path.join(word_dict_path, 'utable.npy')
        #self.params = np.load(path_params, encoding='latin1')
        self.embed = fastText.load_model(args.dict)
        #self.dico = _load_dictionary(word_dict_path)

    def __getitem__(self, index, raw=False):
        path = self.imList[int(index)]
        target = self.capList[int(index)]
        img = Image.open(path).convert('RGB')
            
        img = self.transform(img)
        
        target = encode_sentence_fasttext(target, self.embed, False)
        
        return img, target

    def __len__(self):
        return len(self.imList)


class Multi30k(data.Dataset):
    
    def __init__(self, sset="train", image_dir="/data/flickr30k/flickr30k_images", split_dir="data/image_splits", tok_dir="data/tok", lang='en', transform=None):
        self.transform = transform
        self.imList = []
        self.rootDir = image_dir
        
        if "en" in lang:
            self.engEmb, _, self.engWordsID = _load_vec('data/wiki.multi.en.vec')
        if "fr" in lang:
            self.frEmb, _, self.frWordsID = _load_vec('data/wiki.multi.fr.vec')
        if "de" in lang:
            self.deEmb, _, self.deWordsID = _load_vec('data/wiki.multi.de.vec')
        if "cs" in lang:
            self.csEmb, _, self.csWordsID = _load_vec('data/wiki.multi.cs.vec')
        
        
        self.captions = []
        
        if "train" in sset:
            imFile = os.path.join(split_dir, "train.txt")
            if "fr" in lang:
                for i, line in enumerate(open(os.path.join(tok_dir, "train.lc.norm.tok.fr"))):
                    self.captions.append( (line.rstrip(), 'fr', i) )
            if "en" in lang:
                for i, line in enumerate(open(os.path.join(tok_dir, "train.lc.norm.tok.en"))):
                    self.captions.append( (line.rstrip(), 'en', i) )
            if "de" in lang:
                for i, line in enumerate(open(os.path.join(tok_dir, "train.lc.norm.tok.de"))):
                    self.captions.append( (line.rstrip(), 'de', i) )
            if "cs" in lang:
                for i, line in enumerate(open(os.path.join(tok_dir, "train.lc.norm.tok.de"))):
                    self.captions.append( (line.rstrip(), 'cs', i) )
            
        elif "val" in sset:
            imFile = os.path.join(split_dir, "val.txt")
            
            if "fr" in lang:
                for i, line in enumerate(open(os.path.join(tok_dir, "val.lc.norm.tok.fr"))):
                    self.captions.append( (line.rstrip(), 'fr', i) )
            if "en" in lang:
                for i, line in enumerate(open(os.path.join(tok_dir, "val.lc.norm.tok.en"))):
                    self.captions.append( (line.rstrip(), 'en', i) )
            if "de" in lang:
                for i, line in enumerate(open(os.path.join(tok_dir, "val.lc.norm.tok.de"))):
                    self.captions.append( (line.rstrip(), 'de', i) )
            if "cs" in lang:
                for i, line in enumerate(open(os.path.join(tok_dir, "val.lc.norm.tok.de"))):
                    self.captions.append( (line.rstrip(), 'cs', i) )
            
        else:
            imFile = os.path.join(split_dir, "test_2016_flickr.txt")
            
            if "fr" in lang:
                for i, line in enumerate(open(os.path.join(tok_dir, "test_2016_flickr.lc.norm.tok.fr"))):
                    self.captions.append( (line.rstrip(), 'fr', i) )
            if "en" in lang:
                for i, line in enumerate(open(os.path.join(tok_dir, "test_2016_flickr.lc.norm.tok.en"))):
                    self.captions.append( (line.rstrip(), 'en', i) )
            if "de" in lang:
                for i, line in enumerate(open(os.path.join(tok_dir, "test_2016_flickr.lc.norm.tok.de"))):
                    self.captions.append( (line.rstrip(), 'de', i) )
            if "cs" in lang:
                for i, line in enumerate(open(os.path.join(tok_dir, "test_2016_flickr.lc.norm.tok.de"))):
                    self.captions.append( (line.rstrip(), 'cs', i) )
            
        
        for line in open(imFile):
            imName = line.rstrip()
            self.imList.append(os.path.join(image_dir,imName))
            
            
    def __len__(self):
        return len(self.captions)
            
    def __getitem__(self, index):    
        caption, lang, imId = self.captions[index]
        if lang == 'fr':
            cap = encode_sentence(caption, self.frEmb, self.frWordsID, tokenize=False)
        elif lang == 'en':
            cap = encode_sentence(caption, self.engEmb, self.engWordsID, tokenize=False)
        elif lang == 'de':
            cap = encode_sentence(caption, self.deEmb, self.deWordsID, tokenize=False)
        elif lang == 'cs':
            cap = encode_sentence(caption, self.csEmb, self.csWordsID, tokenize=False)
        else:
            print("Unknown language : ", lang)
            return None
        
        im = self.transform(Image.open(self.imList[imId]))
        
        #return caption, cap
        #return self.imList[imId], caption
        return im, cap
        
            
        
    


class FileDataset(data.Dataset):

    def __init__(self, img_dir_paths, imgs=None, transform=None):
        self.transform = transform
        self.root = img_dir_paths
        self.imgs = imgs or [os.path.join(img_dir_paths, f) for f in os.listdir(img_dir_paths) if re.match(r'.*\.jpg', f)]

    def __getitem__(self, index):

        img = Image.open(self.imgs[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def get_image_list(self):
        return self.imgs

    def __len__(self):
        return len(self.imgs)




class DoubleDataset(data.Dataset):
    def __init__(self, d1, d2):
        self.d1 = d1
        self.d2 = d2
        
    def __len__(self):
        return len(self.d1) + len(self.d2)
    
    def __getitem__(self, index):
        if index >= len(self.d1):
            return self.d2[index-len(self.d1)]
        return self.d1[index]


"""
class TextDataset(data.Dataset):
    def __init__(self, text_path, word_dict_path=path["WORD_DICT"]):
        with open(text_path) as f:
            lines = f.readlines()
        self.sent_list = [line.rstrip('\n') for line in lines]
        path_params = os.path.join(word_dict_path, 'utable.npy')
        self.params = np.load(path_params, encoding='latin1')
        self.dico = _load_dictionary(word_dict_path)
    def __getitem__(self, index):
        caption = self.sent_list[index]
        caption = encode_sentence(caption, self.params, self.dico)
        return caption
    def __len__(self):
        return len(self.sent_list)
"""



"""
class VgCaptions(data.Dataset):
    def __init__(self, coco_root=path["COCO_ROOT"], vg_path_ann=path["VG_ANN"], path_vg_img=path["VG_IMAGE"], coco_json_file_path=path["COCO_RESTVAL_SPLIT"], word_dict_path=path["WORD_DICT"], image=True, transform=None):
        self.transform = transform
        self.image = image
        path_params = os.path.join(word_dict_path, 'utable.npy')
        self.params = np.load(path_params, encoding='latin1')
        self.dico = _load_dictionary(word_dict_path)
        self.path_vg_img = path_vg_img
        ids = vg.get_all_image_data(vg_path_ann)
        regions = vg.get_all_region_descriptions(vg_path_ann)
        annFile = os.path.join(coco_root, "annotations/captions_val2014.json")
        coco = COCO(annFile)
        ids_val_coco = list(coco.imgs.keys())
        # Uncomment following bloc to evaluate only on validation set from Rest/Val split
        # with open(coco_json_file_path, 'r') as f: # coco_json_file_path = "/home/wp01/users/engilbergem/dev/trunk/CPLApplications/deep/PytorchApplications/coco/dataset.json"
        #     datas = json.load(f)
        # ids_val_coco = [x['cocoid'] for x in datas["images"] if x["split"] == "val"]  # list(coco.imgs.keys())
        self.data = [x for x in zip(ids, regions) if x[0].coco_id in ids_val_coco]
        self.imgs_paths = [x[0].id for x in self.data]
        self.nb_regions = [len([x.phrase for x in y[1]])
                           for y in self.data]
        self.captions = [x.phrase for y in self.data for x in y[1]]
    def __getitem__(self, index, raw=False):
        if self.image:
            id_vg = self.data[index][0].id
            img = Image.open(os.path.join(self.path_vg_img,
                                          str(id_vg) + ".jpg")).convert('RGB')
            if raw:
                return img
            if self.transform is not None:
                img = self.transform(img)
            return img
        else:
            target = self.captions[index]
            #  If the caption is incomplete we set it to zero
            if len(target) < 3:
                target = torch.FloatTensor(1, 620)
            else:
                target = encode_sentence(target, self.params, self.dico)
            return target
    def __len__(self):
        if self.image:
            return len(self.data)
        else:
            return len(self.captions)
"""

"""
class CocoSemantic(data.Dataset):
    def __init__(self, coco_root=path["COCO_ROOT"], word_dict_path=path["WORD_DICT"], transform=None):
        self.coco_root = coco_root
        annFile = os.path.join(coco_root, "annotations/instances_val2014.json")
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        path_params = os.path.join(word_dict_path, 'utable.npy')
        params = np.load(path_params, encoding='latin1')
        dico = _load_dictionary(word_dict_path)
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        # repeats category with plural version
        categories_sent = [cat['name'] + " " + cat['name'] + "s" for cat in self.categories]
        self.categories_w2v = [encode_sentence(cat, params, dico, tokenize=True) for cat in categories_sent]
    def __getitem__(self, index, raw=False):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        target = dict()
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.coco_root, "images/val2014/", path)).convert('RGB')
        img_size = img.size
        for ann in anns:
            key = [cat['name'] for cat in self.categories if cat['id'] == ann["category_id"]][0]
            if key not in target:
                target[key] = list()
            if type(ann['segmentation']) != list:
                if type(ann['segmentation']['counts']) == list:
                    rle = maskUtils.frPyObjects(
                        [ann['segmentation']], img_size[0], img_size[1])
                else:
                    rle = [ann['segmentation']]
                target[key] += [("rle", rle)]
            else:
                target[key] += ann["segmentation"]
        if raw:
            return path, target
        if self.transform is not None:
            img = self.transform(img)
        return img, img_size, target
    def __len__(self):
        return len(self.ids)
"""

def main(batch_size=32, workers=4 ):
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
        
    
    
    coco_data_train = Multi30k(sset="trainrv", transform=prepro)
    coco_data_val = Multi30k(sset="val", transform=prepro_val)
    
    train_loader = DataLoader(coco_data_train, batch_size=1, shuffle=True, drop_last=False,
                              num_workers=workers, collate_fn=collate_fn_padded, pin_memory=True)
    val_loader = DataLoader(coco_data_val, batch_size=batch_size, shuffle=False,
                            num_workers=workers, collate_fn=collate_fn_padded, pin_memory=True, drop_last=True)
    
    
    print(coco_data_train[0][0])
    print(coco_data_train[0][1].shape)
    
    for i, b in enumerate(train_loader):
        print(b[1].shape)
        sys.exit(0)
        if np.isnan(b[0]).any():
                print("------------------")
                print("NaN found in image")
                print("------------------")
        if np.isnan(b[1]).any():
                print("------------------")
                print("NaN found in caption")
                print("------------------")
        print(b[1].shape)
        if b[1].sum() == 0:
            __import__('ipdb');ipdb.set_trace()
            
            print("Found caption at 0")
            print("cap :", coco_data_train[i])
                
        print("%2.2f"% (i/len(train_loader)*100), '\%', end='\r')
    
    for i, b in enumerate(val_loader):
        if np.isnan(b[0]).any():
                print("------------------")
                print("NaN found in image")
                print("------------------")
        if np.isnan(b[1]).any():
                print("------------------")
                print("NaN found in caption")
                print("------------------")
        print("%2.2f"% (i/len(val_loader)*100), '\%', end='\r')
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument("-bs", "--batch_size", help="The size of the batches", type=int, default=4)
    parser.add_argument("-w", dest="workers", help="Nb workers", default=4, type=int)

    args=parser.parse_args()

    main(args.batch_size, args.workers)


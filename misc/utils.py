"""
****************** COPYRIGHT AND CONFIDENTIALITY INFORMATION ******************
Copyright (c) 2018 [Thomson Licensing]
All Rights Reserved
This program contains proprietary information which is a trade secret/business \
secret of [Thomson Licensing] and is protected, even if unpublished, under \
applicable Copyright laws (including French droit d'auteur) and/or may be \
subject to one or more patent(s).
Recipient is to retain this program in confidence and is not permitted to use \
or make copies thereof other than as permitted in a written agreement with \
[Thomson Licensing] unless otherwise expressly allowed by applicable laws or \
by [Thomson Licensing] under express agreement.
Thomson Licensing is a company of the group TECHNICOLOR
*******************************************************************************
This scripts permits one to reproduce training and experiments of:
    Engilberge, M., Chevallier, L., Pérez, P., & Cord, M. (2018, April).
    Finding beans in burgers: Deep semantic-visual embedding with localization.
    In Proceedings of CVPR (pp. 3984-3993)

Author: Martin Engilberge
"""

import os

import nltk
import pickle
import torch
import io
import numpy as np

from nltk.tokenize import word_tokenize
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence

import fastText


class AverageMeter(object):

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


class Namespace:
    """ Namespace class to manually instantiate joint_embedding model """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _load_dictionary(dir_st):
    path_dico = os.path.join(dir_st, 'dictionary.txt')
    if not os.path.exists(path_dico):
        print("Invalid path no dictionary found")
    with open(path_dico, 'r') as handle:
        dico_list = handle.readlines()
    dico = {word.strip(): idx for idx, word in enumerate(dico_list)}
    return dico


def load_vec(emb_path):
    """
        Load FastText model .vec
        Returns : embeddings (matrix index -> embedding), id2word ( index -> word ) and word2id ( word -> index)
    """
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


def preprocess(text):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = sent_detector.tokenize(text)
    result = list()
    for s in sents:
        tokens = word_tokenize(s)
        result.append(tokens)
    return result

def fr_preprocess(sentence):
    return word_tokenize(sentence)


def flatten(l):
    return [item for sublist in l for item in sublist]


def encode_sentences(sents, embed, dico):
    sents_list = list()
    for sent in sents:
        sent_tok = preprocess(sent)[0]
        sent_in = Variable(torch.FloatTensor(1, len(sent_tok), 300))
        for i, w in enumerate(sent_tok):
            try:
                sent_in.data[0, i] = torch.from_numpy(embed[dico[w]])
            except KeyError:
                sent_in.data[0, i] = torch.from_numpy(embed[dico["UNK"]])

        sents_list.append(sent_in)
    return sents_list


def encode_sentence(sent, embed, dico, tokenize=True):
    """
        Encode one sentence but sliting spaces
        Input : sentence (string), embeddings matrix, dicionary (word -> index in the matrix)
        Output : embedding of each word in the sentence
    """
    sent = sent.strip().lower()
    if tokenize:
        sent_tok = word_tokenize(sent)
    else:
        sent_tok = sent.split(' ')
        
    embed_size = len(embed[0])
    sent_in = torch.FloatTensor(len(sent_tok), embed_size)

    for i, w in enumerate(sent_tok):
        w = w.strip()
        if '-' in w:
            w = w.split('-')[-1]
        try:
            sent_in[i, :embed_size] = torch.from_numpy(embed[dico[w]])
        except KeyError:
            sent_in[i, :embed_size] = torch.from_numpy(embed[0])
            #sent_in[i, :300] = torch.from_numpy(embed.get_word_vector("unk"))

    return sent_in

def encode_sentence_fasttext(sent, embed, tokenize=True, french=False):
    """
        Encode sentence with fasttext model
        Input : sentence to encode and fasttext model
        return embedding for each word in the sentence
    """
    if tokenize:
        if french:
            sent_tok = fr_preprocess(sent)
        else:
            sent_tok = preprocess(sent)[0]
    else:
        sent_tok = sent
    embed_size = embed.get_dimension()

    sent_in = torch.FloatTensor(len(sent_tok), embed_size)

    for i, w in enumerate(sent_tok):
        sent_in[i, :embed_size] = torch.from_numpy(embed.get_word_vector(w))
    return sent_in


def encode_sentence_vec(sent, embed, word2id, tokenize=True):
    if tokenize:
        sent_tok = preprocess(sent)[0]
    else:
        sent_tok = sent

    sent_in = torch.FloatTensor(len(sent_tok), len(embed[0]))

    for i, w in enumerate(sent_tok):
        sent_in[i, :embed_size] = torch.from_numpy(embed[word2id[w]])
    return sent_in


def save_checkpoint(state, is_best, model_name, epoch):
    if is_best:
        print("saving best model...")
        torch.save(state, './weights/best_' + model_name + ".pth.tar")
    #else:
    #    torch.save(state, './weights/epoch_'+str(epoch)+'_'+model_name+".pth.tar")

def log_epoch(logger, epoch, train_loss, val_loss, lr, batch_train, batch_val, data_train, data_val, recall):
    logger.add_scalar('Loss/Train', train_loss, epoch)
    logger.add_scalar('Loss/Val', val_loss, epoch)
    logger.add_scalar('Learning/Rate', lr, epoch)
    logger.add_scalar('Learning/Overfitting', val_loss / train_loss, epoch)
    logger.add_scalar('Time/Train/Batch Processing', batch_train, epoch)
    logger.add_scalar('Time/Val/Batch Processing', batch_val, epoch)
    logger.add_scalar('Time/Train/Data loading', data_train, epoch)
    logger.add_scalar('Time/Val/Data loading', data_val, epoch)
    logger.add_scalar('Recall/Val/CapRet/R@1', recall[0][0], epoch)
    logger.add_scalar('Recall/Val/CapRet/R@5', recall[0][1], epoch)
    logger.add_scalar('Recall/Val/CapRet/R@10', recall[0][2], epoch)
    logger.add_scalar('Recall/Val/CapRet/MedR', recall[2], epoch)
    logger.add_scalar('Recall/Val/ImgRet/R@1', recall[1][0], epoch)
    logger.add_scalar('Recall/Val/ImgRet/R@5', recall[1][1], epoch)
    logger.add_scalar('Recall/Val/ImgRet/R@10', recall[1][2], epoch)
    logger.add_scalar('Recall/Val/ImgRet/MedR', recall[3], epoch)


def collate_fn_padded(data):
    images, captions = zip(*data)

    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in captions]
    targets = pad_sequence(captions, batch_first=True)

    return images, targets, lengths



def collate_fn_cap_index(data):
    captions, indices = zip(*data)
    captions = pad_sequence(captions, batch_first=True)
    return captions, indices

def collate_fn_cap_padded(data):
    captions = data

    lengths = [len(cap) for cap in captions]
    targets = pad_sequence(captions, batch_first=True)

    return targets, lengths


def collate_fn_semseg(data):
    images, size, targets = zip(*data)
    images = torch.stack(images, 0)

    return images, size, targets


def collate_fn_img_padded(data):
    images = data
    images = torch.stack(images, 0)

    return images


def load_obj(path):
    with open(os.path.normpath(path + '.pkl'), 'rb') as f:
        return pickle.load(f)


def save_obj(obj, path):
    with open(os.path.normpath(path + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def checkDataset(datasetFile):
    imList = []
    capList = []
    with open(datasetFile) as f:
        for i, line in enumerate(f):
            cap = line.split('\t')[-1]
            #imList.append(os.path.join(root_dir, im+'.jpg'))
            if 1 < len(cap.split(' ')) < 20:
                capList.append(cap)
    print("Read :", len(capList), "images and caption")
    embed = fastText.load_model('data/wiki.fr.bin')
    ERROR = 0
    m = 0
    for cap in capList:
        ft = encode_sentence_fasttext(cap, embed, True, True)    
        if len(ft) < 1:
            print("Small : ", cap)
            ERROR = 1
        if len(cap) > m:
            m = len(cap)
            mcap = cap
    return ERROR, m, mcap
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

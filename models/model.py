import torch
import torch.nn as nn

from utils.config import path
from models.weldonModel import ResNet_weldon
from sru import SRU
import torch.nn.init as weight_init


class SruEmb(nn.Module):
    def __init__(self, nb_layer, dim_in, dim_out, dropout=0.25):
        super(SruEmb, self).__init__()
        self.dim_out = dim_out
        self.rnn = SRU(dim_in, dim_out, num_layers=nb_layer,
                       dropout=dropout, rnn_dropout=dropout,
                       use_tanh=True, has_skip_term=True,
                       v1=True, rescale=False)
         
    def _select_last(self, x, lengths):
        batch_size = x.size(0)
        mask = x.data.new().resize_as_(x.data).fill_(0)
        for i in range(batch_size):
            mask[i][lengths[i] - 1].fill_(1)
        x = x.mul(mask)
        x = x.sum(1, keepdim=True).view(batch_size, self.dim_out)
        return x
        
    def _process_lengths(self, input):
        max_length = input.size(1)
        lengths = [max_length - input.data[i].eq(0).sum(1, keepdim=True).squeeze() for i in range(input.shape[0])]
        #lengths = list(
        #    max_length - input.data.eq(0).sum(1, keepdim=True).squeeze())
        return lengths
        
    def forward(self, input, lengths=None):
        if lengths is None:
            lengths = self._process_lengths(input)
        x = input.permute(1, 0, 2)
        x, hn = self.rnn(x)
        x = x.permute(1, 0, 2)
        if lengths:
            x = self._select_last(x, lengths)
        return x


class GruEmb(nn.Module):
    def __init__(self, nb_layer, dim_in, dim_out, dropout=0.25):
        super(GruEmb, self).__init__()
        self.dim_out = dim_out
        self.rnn = nn.GRU(dim_in, dim_out, num_layers=nb_layer,
                       dropout=dropout,
                       batch_first=True)
    def forward(self, input):
        x, hn = self.rnn(input)
        return hn[-1] #select last hidden (or last output)



class img_embedding(nn.Module):

    def __init__(self, args=None):
        super(img_embedding, self).__init__()
        if args is None:
            model_weldon2 = ResNet_weldon(None, pretrained=True, weldon_pretrained_path=path["WELDON_CLASSIF_PRETRAINED"])
        else:
            model_weldon2 = ResNet_weldon(args, pretrained=True, weldon_pretrained_path=path["WELDON_CLASSIF_PRETRAINED"])
        self.base_layer = nn.Sequential(*list(model_weldon2.children())[:-1])
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.base_layer(x)
        x = x.view(x.size()[0], -1)

        return x

    def get_activation_map(self, x):
        x = self.base_layer[0](x)
        act_map = self.base_layer[1](x)
        act = self.base_layer[2](act_map)
        return act, act_map


class joint_embedding(nn.Module):

    def __init__(self, args=None):
        super(joint_embedding, self).__init__()
        if not args is None:
            self.img_emb = img_embedding(args)
            self.cap_emb = SruEmb(args.sru, args.embed_size, args.dimemb)
            #self.cap_emb = SruEmb(args.sru, 300, args.dimemb)
            #self.cap_emb = torch.nn.DataParallel(GruEmb(args.sru, 300, args.dimemb))
            
            self.dropout = torch.nn.Dropout(p=0.5)
            self.fc = nn.Linear(2400, args.dimemb, bias=True)
        else:
            self.img_emb = img_embedding()
            self.cap_emb = SruEmb(4, 300, 2400)
            self.dropout = torch.nn.Dropout(p=0.5)
            self.fc = nn.Linear(2400, 2400, bias=True)

    def forward(self, imgs, caps, lengths):
        if imgs is not None:
            x_imgs = self.img_emb(imgs)
            x_imgs = self.dropout(x_imgs)
            x_imgs = self.fc(x_imgs)

            x_imgs = x_imgs / torch.norm(x_imgs, 2, dim=1, keepdim=True).expand_as(x_imgs)
        else:
            x_imgs = None

        if caps is not None:
            x_caps = self.cap_emb(caps, lengths=lengths)
            #x_caps = self.cap_emb(caps)
            x_caps = x_caps / torch.norm(x_caps, 2, dim=1, keepdim=True).expand_as(x_caps)
        else:
            x_caps = None

        return x_imgs, x_caps

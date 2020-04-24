import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from .Model import Model

mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class SVCNN(Model):

    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='vgg11', KNU_data=True, use_encdec=False, encdec_name='alexnet', encdim=4096):
        super(SVCNN, self).__init__(name)

        if KNU_data:
            self.classnames = ['BlindFlange', 'Cross', 'Elbow 90', 'Elbow non 90', 'Flange', 'Flange WN',
                               'Olet', 'OrificeFlange', 'Pipe', 'Reducer CONC', 'Reducer ECC',
                               'Reducer Insert', 'Safety Valve', 'Strainer', 'Tee', 'Tee RED',
                               'Valve', 'Wye']
        else:
            self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                               'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                               'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                               'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                               'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_encdec = use_encdec
        self.encdec_name = encdec_name
        self.encdim = encdim
        self.use_resnet = cnn_name.startswith('resnet')
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

        if self.use_encdec:
            if self.encdec_name == 'simpleNet':
                self.encnet = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=5, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(64, 128, kernel_size=5, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(128, 256, kernel_size=5, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                )
                self.decnet = nn.Sequential(
                    nn.ConvTranspose2d(256, 128, kernel_size=7, stride= 2, padding=2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(128, 64, kernel_size=7, stride= 2, padding=2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(64, 3, kernel_size=6, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                )

            if self.encdec_name == 'alexnet':
                self.encnet = models.alexnet().features
                # (6x6x256 to 224x224x3)
                self.decnet = nn.Sequential(
                    nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(256, 384, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(384, 192, kernel_size=5, stride=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(192, 64, kernel_size=9, stride=3, padding=2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(64, 3, kernel_size=10, stride=4, padding=3),
                    nn.ReLU(inplace=True)
                )
            if self.encdec_name == 'vgg11':
                self.encnet = models.alexnet().features
                # (6x6x256 to 224x224x3)
                self.decnet = nn.Sequential(
                    nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(256, 384, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(384, 192, kernel_size=5, stride=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(192, 64, kernel_size=9, stride=3, padding=2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(64, 3, kernel_size=10, stride=4, padding=3),
                    nn.ReLU(inplace=True)
                )

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512,nclasses)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512,nclasses)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048,nclasses)
        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11(pretrained=self.pretraining).features
                self.net_2 = models.vgg11(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier
            
            self.net_2._modules['6'] = nn.Linear(4096,nclasses)

    def forward(self, x):
        if self.use_encdec:
            x = self.encnet(x)
            x = self.decnet(x)

        if self.use_resnet:
            return self.net(x)
        else:
            y = self.net_1(x)
            return self.net_2(y.view(y.shape[0],-1))


class MVCNN(Model):

    def __init__(self, name, model, nclasses, cnn_name, num_views, KNU_data, use_encdec, encdec_name, encdim, use_dataparallel):
        super(MVCNN, self).__init__(name)

        if KNU_data:
            self.classnames = ['BlindFlange', 'Cross', 'Elbow 90', 'Elbow non 90', 'Flange', 'Flange WN',
                               'Olet', 'OrificeFlange', 'Pipe', 'Reducer CONC', 'Reducer ECC',
                               'Reducer Insert', 'Safety Valve', 'Strainer', 'Tee', 'Tee RED',
                               'Valve', 'Wye']
        else:
            self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                               'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                               'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                               'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                               'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

        self.nclasses = nclasses
        self.num_views = num_views
        self.use_encdec = use_encdec
        self.encdec_name = encdec_name
        self.encdim = encdim
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_encdec:
            if use_dataparallel == True:
                self.encnet = model.modules.encnet
                self.decnet = model.modules.decnet
            else:
                self.encnet = model.encnet
                self.decnet = model.decnet

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
        else:
            if use_dataparallel == True:
                self.net_1 = model.module.net_1
                self.net_2 = model.module.net_2
            else:
                self.net_1 = model.net_1
                self.net_2 = model.net_2

    def forward(self, x):
        if self.use_encdec:
            x = self.encnet(x)
            x = self.decnet(x)

        y = self.net_1(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1]))#(8,12,512,7,7)
        return self.net_2(torch.max(y,1)[0].view(y.shape[0],-1))


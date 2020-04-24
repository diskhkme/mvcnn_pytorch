import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse

from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from models.MVCNN import MVCNN, SVCNN

from tools.focalloss import *

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="Vgg11_Seg_Stage2_Only_EncDec_simple")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=4)# it will be *12 images in each batch for mvcnn
parser.add_argument("-num_models", type=int, help="number of models per class", default=1000)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="vgg11")
parser.add_argument("-num_views", type=int, help="number of views", default=12)
# ROOTPATH = 'D:/KHK/Data/SegmentedPointCloud/ModelNet40/ModelNet40_Depth' # ModelNet Depth
ROOTPATH = 'D:/KHK/Data/SegmentedPointCloud/SegmentedPointCloud/224_224_Depth' # 기존 Depth
# ROOTPATH = 'D:/KHK/Data/SegmentedPointCloud/SegmentedPointCloud/224_224_Depth_White' # ModelNet과 동일하게 배경 흰 Depth
parser.add_argument("-train_path", type=str, default=ROOTPATH+"/*/train")
parser.add_argument("-val_path", type=str, default=ROOTPATH+"/*/test")

parser.add_argument("-KNU_Data", type=bool, default=True)
parser.add_argument("-loss_type", type=str, default='cross') # 'focal_loss' or else

parser.add_argument("-use_encdec", type=bool, default=True)
parser.add_argument("-encdec_name", type=str, default='simpleNet')
parser.add_argument("-encdim", type=int, default=4096) # 현재 사용 안함
parser.add_argument("-use_dataparallel", type=bool, default=False) # 현재 사용 안함

parser.add_argument("-pixel_augmentation", type=bool, default=False)


parser.set_defaults(train=False)

def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)

if __name__ == '__main__':

    args = parser.parse_args()

    if args.KNU_Data == True:
        nclasses = 18
    else:
        nclasses = 40

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_dataparallel = args.use_dataparallel
    pretraining = not args.no_pretraining
    log_dir = args.name
    create_folder(args.name)
    config_f = open(os.path.join(log_dir, 'config.json'), 'w')
    json.dump(vars(args), config_f)
    config_f.close()

    # STAGE 1
    # log_dir = args.name+'_stage_1'
    # create_folder(log_dir)
    cnet = SVCNN(args.name, nclasses=nclasses, pretraining=pretraining, cnn_name=args.cnn_name,KNU_data=args.KNU_Data,
                 use_encdec=args.use_encdec, encdec_name=args.encdec_name, encdim=args.encdim)
    #
    # if use_dataparallel:
    #     cnet = nn.DataParallel(cnet)
    #     cnet.to(device)
    #
    # optimizer = optim.Adam(cnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #
    n_models_train = args.num_models*args.num_views

    # train_dataset = SingleImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views,KNU_data=args.KNU_Data)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    #
    # val_dataset = SingleImgDataset(args.val_path, scale_aug=False, rot_aug=False, test_mode=True,KNU_data=args.KNU_Data)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    # print('num_train_files: '+str(len(train_dataset.filepaths)))
    # print('num_val_files: '+str(len(val_dataset.filepaths)))
    #
    # if(args.loss_type == 'focal_loss'):
    #     focal_loss = FocalLoss(gamma=2, alpha=0.25)
    #     trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer, focal_loss, 'svcnn', log_dir, num_views=1, nClasses=nclasses)
    # else:
    #     trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'svcnn', log_dir,
    #                               num_views=1, nClasses=nclasses)
    # trainer.train(30, use_dataparallel)

    # STAGE 2
    ### -----------------------stage 2부터 시작할 때만 필요한 코드!---------------------------------------------
    # if use_dataparallel == True:
    #     cnet.module.load_state_dict(torch.load("Vgg11_Seg_white_stage_1/Vgg11_Seg_white/model-00022.pth"))
    #     cnet.module.eval()
    # else:
    #     cnet.load_state_dict(torch.load("Vgg11_Seg_white_stage_1/Vgg11_Seg_white/model-00022.pth"))
    #     cnet.eval()
    ### -----------------------stage 2부터 시작할 때만 필요한 코드!---------------------------------------------

    log_dir = args.name+'_stage_2'
    create_folder(log_dir)
    cnet_2 = MVCNN(args.name, cnet, nclasses=nclasses, cnn_name=args.cnn_name, num_views=args.num_views,KNU_data=args.KNU_Data,
                 use_encdec=args.use_encdec, encdec_name=args.encdec_name, encdim=args.encdim, use_dataparallel=use_dataparallel)
    del cnet

    ### -----------------------stage 2의 30epoch 이상 학습시 필요한 코드!---------------------------------------------
    # if use_dataparallel == True:
    #     cnet_2.module.load_state_dict(torch.load("Vgg11_Seg_white_stage_2/Vgg11_Seg_white/model-00027.pth"))
    #     cnet_2.module.eval()
    # else:
    #     cnet_2.load_state_dict(torch.load("Vgg11_Seg_white_stage_2/Vgg11_Seg_white/model-00027.pth"))
    #     cnet_2.eval()
    ### -------------------------------------------------------------------------------------------------------------

    if use_dataparallel:
        cnet_2 = nn.DataParallel(cnet_2)
        cnet_2.to(device)

    optimizer = optim.Adam(cnet_2.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    
    train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views,KNU_data=args.KNU_Data,pixel_augmentation=args.pixel_augmentation)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)# shuffle needs to be false! it's done within the trainer

    val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, num_views=args.num_views,KNU_data=args.KNU_Data,pixel_augmentation=args.pixel_augmentation)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))

    if(args.loss_type == 'focal_loss'):
        focal_loss = FocalLoss(gamma=2, alpha=0.25)
        trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, focal_loss, 'mvcnn', log_dir, num_views=args.num_views, nClasses=nclasses)
    else:
        trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'mvcnn', log_dir, num_views=args.num_views, nClasses=nclasses)

    trainer.train(30, use_dataparallel)



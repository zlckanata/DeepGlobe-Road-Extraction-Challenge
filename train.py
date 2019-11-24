import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
import sys 


import cv2
import os
import numpy as np

from time import time

from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder

SHAPE = (1024,1024)
ROOT = '/content/DeepGlobe-Road-Extraction-Challenge/dataset/train/'
imagelist = filter(lambda x: x.find('sat')!=-1, os.listdir(ROOT))
trainlist = [x[:-8] for x in imagelist] #    map(lambda x: x[:-8], imagelist)
NAME = 'log01_dink34'
BATCHSIZE_PER_CARD = 4
solver = MyFrame(DinkNet34, dice_bce_loss, 2e-4)

batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

dataset = ImageFolder(trainlist, ROOT)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=4)
argumentList = sys.argv 
mylog = open('logs/'+NAME+'.log','w')
tic = time()
no_optim = 0
total_epoch = 300
train_epoch_best_loss = 100
startt = 0
print(argumentList)
if(len(argumentList) == 2):
    path = argumentList[1]
    checkpoint = torch.load(path)
    solver.load_state_dict(checkpoint['model_state_dict'])
    solver.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    startt = checkpoint['epoch']
    losss = checkpoint['loss']
for epoch in range(startt, total_epoch + 1):
	#solver.load("/content/gdrive/My Drive/model.pt", epoch)
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    for img, mask in data_loader_iter:
        solver.set_input(img, mask)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss
    train_epoch_loss /= len(data_loader_iter)
    print('********', file = mylog)
    print('epoch:',epoch,'    time:',int(time()-tic), file = mylog)
    print('train_loss:',train_epoch_loss, file = mylog)
    print( 'SHAPE:',SHAPE, file = mylog)
    print('********')
    print('epoch:',epoch,'    time:',int(time()-tic))
    print('train_loss:',train_epoch_loss)
    print('SHAPE:',SHAPE)
    
    if(epoch % 10 == 0):
    	solver.save("/content/gdrive/My Drive/model.pt", epoch,train_epoch_loss)
    	checkpoint = torch.load("/content/gdrive/My Drive/model.pt")
    	solver.load_state_dict(checkpoint['model_state_dict'])
    	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    	epoch = checkpoint['epoch']
    	train_epoch_loss = checkpoint['loss']
    	
    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save('weights/'+NAME+'.th')
    if no_optim > 6:
        print('early stop at %d epoch' % epoch, file = mylog)
        print('early stop at %d epoch' % epoch)
        break
    if no_optim > 3:
        if solver.old_lr < 5e-7:
            break
        solver.load('weights/'+NAME+'.th')
        solver.update_lr(5.0, factor = True, mylog = mylog)
    mylog.flush()
    
print('Finish!', file = mylog)
print('Finish!')
mylog.close()

import torch
import torch.nn as nn
from torch.autograd import Variable as V

import cv2
import numpy as np
# Creating a generic Framework class.
class MyFrame():
    def __init__(self, net, loss, lr=2e-4, evalmode = False):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        #self.optimizer = torch.optim.RMSprop(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()

    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id

    def test_one_img(self, img):
        pred = self.net.forward(img)

        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()
        return mask


    def test_batch(self):
        self.forward(volatile=True)
        mask =  self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0

        return mask, self.img_id

    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32)/255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())

        mask = self.net.forward(img).squeeze().cpu().data.numpy()#.squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0

        return mask

    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)

    ## OPTIMIZE: using the mentioned optimizer
    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        loss = self.loss(self.mask, pred)
        loss.backward()
        self.optimizer.step()
        return loss.data

    # Saving the complete model
    # No need to use this function
    def save_full(self, path):
        torch.save(self.net, path)

    # Method to save the model for general checkpoint.
    # Model_state_dict contanis all the required weights.
    # Epoch is the iteration number
    # Loss value is saved too.
    # Optimizer state dict saves all the gradients of the model
    def save(self, path, epoch,losss):
        torch.save({
		'epoch' : epoch,
		'model_state_dict' : self.net.state_dict(),
		'optimizer_state_dict' : self.optimizer.state_dict(),
		'loss' : losss}, path)

    def load(self, path):
	    checkpoint = torch.load(path)
	    self.net.load_state_dict(checkpoint['model_state_dict'])
	    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #Method to update the Learning rate
    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print('update learning rate: %f -> %f' % (self.old_lr, new_lr), file = mylog)
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr

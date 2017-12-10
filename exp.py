from __future__ import print_function
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
from util import load_flo, load_img, vis_img, show, show_cv2
import time
from Warp import Warp
import torch
import cv2
from torchvision.utils import make_grid

# load data
img1 = load_img('data/img1.ppm')
img2 = load_img('data/img2.ppm')

flow = load_flo('data/flow.flo')
flow = np.transpose(flow, (2, 0, 1))
flow = torch.from_numpy(flow)
flow.unsqueeze_(0)

warp = Warp()
start = time.time()
fake_img1 = warp(Variable(img2, requires_grad=False), Variable(flow, requires_grad=True))
# show_cv2(make_grid(fake_img1.data, nrow=1))
np_img = fake_img1.data[0].numpy().transpose((1, 2, 0)).astype(np.uint8)
np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
# cv2.imshow('warp', np_img)
# cv2.waitKey()
cv2.imwrite('warp_result.png', np_img)
# cv2.imshow('fake_img', cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR))
# cv2.waitKey()
# np_img1 = fake_img1.data.numpy()[0].transpose((1, 2, 0)).astype(np.uint8)
# vis_img(np_img1)

# vis_img(fake_img1.byte().data.numpy()[0])

# criterion = nn.L1Loss()
# loss = criterion(fake_img1, Variable(img1))
#
# warp.zero_grad()
# loss.backward()
#
# pass
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
from util import load_flo, load_img, vis_img
import time
from Warp import Warp

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
print('time: {}'.format(time.time() - start))

# vis_img(fake_img1.byte().data.numpy()[0])

criterion = nn.L1Loss()
loss = criterion(fake_img1, Variable(img1))

warp.zero_grad()
loss.backward()

print()
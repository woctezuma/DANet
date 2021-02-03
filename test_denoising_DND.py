#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-07-10 14:38:39

'''
In this demo, we only test the model on one image of SIDD validation dataset.
The full validation dataset can be download from the following website:
    https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php
'''

import argparse
import torch
import torch.nn as nn
import sys
from networks import UNetD
from scipy.io import loadmat
from skimage import img_as_float32, img_as_ubyte
from matplotlib import pyplot as plt
from utils import PadUNet
from PIL import Image as pil_image

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GDANet+',
                                       help="Model selection: GDANet or GDANet+, (default:GDANet+)")
parser.add_argument('--input_fname', type=str, default='', help="File name of the noisy image")
parser.add_argument('--output_fname', type=str, default='', help="File name of the denoised image")
args = parser.parse_args()


# build the network
dep_U=5
net = UNetD(3, wf=32, depth=dep_U).cuda()

# load the pretrained model
if args.model.lower() == 'gdanet':
    net.load_state_dict(torch.load('./model_states/GDANet.pt', map_location='cpu')['D'])
elif args.model.lower() == 'gdanet+':
    net.load_state_dict(torch.load('./model_states/GDANetPlus_fake025.pt', map_location='cpu'))
else:
    sys.exit('Please input the corrected model')

# read the images
try:
    # Reference: https://stackoverflow.com/a/33507138/
    png = pil_image.open(args.input_fname).convert('RGBA')
    background = pil_image.new('RGBA', png.size, (255,255,255))
    jpg = pil_image.alpha_composite(background, png).convert('RGB')

    im_noisy = jpg
except FileNotFoundError:
    im_noisy = loadmat('./test_data/DND/1.mat')['im_noisy']

# denoising
inputs = torch.from_numpy(img_as_float32(im_noisy).transpose([2,0,1])).unsqueeze(0).cuda()
with torch.autograd.no_grad():
    padunet = PadUNet(inputs, dep_U=5)
    inputs_pad = padunet.pad()
    outputs_pad = inputs_pad - net(inputs_pad)
    outputs = padunet.pad_inverse(outputs_pad)
    outputs.clamp_(0.0, 1.0)

im_denoise = img_as_ubyte(outputs.cpu().numpy()[0,].transpose([1,2,0]))

if len(args.output_fname) > 0:
    pil_image.fromarray(im_denoise).save(args.output_fname)

plt.subplot(1,2,1)
plt.imshow(im_noisy)
plt.title('Noisy Image')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(im_denoise)
plt.title('Denoised Image')
plt.axis('off')
plt.show()


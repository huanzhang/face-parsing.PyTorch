#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2


def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    im = np.array(im)
    origin_im = im.copy().astype(np.uint8)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(
        vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)

    # 换整体背景 - 全黑
    index = np.where(vis_parsing_anno == 0)
    vis_im[index[0], index[1], :] = [0, 0, 0]

    # 换脖子 - 全黑
    index = np.where(vis_parsing_anno == 14)
    vis_im[index[0], index[1], :] = [0, 0, 0]

    # 换衣服 - 全黑
    index = np.where(vis_parsing_anno == 16)
    vis_im[index[0], index[1], :] = [0, 0, 0]

    index = np.where(
        (vis_parsing_anno == 1) |
        (vis_parsing_anno == 2) |
        (vis_parsing_anno == 3) |
        (vis_parsing_anno == 4) |
        (vis_parsing_anno == 5) |
        (vis_parsing_anno == 6) |
        (vis_parsing_anno == 7) |
        (vis_parsing_anno == 8) |
        (vis_parsing_anno == 9) |
        (vis_parsing_anno == 10) |
        (vis_parsing_anno == 11) |
        (vis_parsing_anno == 12) |
        (vis_parsing_anno == 13) |
        (vis_parsing_anno == 15) |
        (vis_parsing_anno == 17))
    vis_im[index[0], index[1], :] = [255, 255, 255]  # 其他全白

    origin = Image.fromarray(origin_im)

    mask = Image.fromarray(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR))
    mask = mask.convert("L")

    empty = Image.new("RGBA", origin.size)
    dst = Image.composite(origin, empty, mask)

    # 保存人像部分，其余透明
    dst.save(save_path[:-4] + '.png')
    rgb_dst = dst.convert("RGB")
    rgb_dst.save(save_path[:-4] + ".jpg", "JPEG",
             quality=100, optimize=True, progressive=True)


def evaluate(respth='./res/out', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth, map_location='cpu'))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            # print(np.unique(parsing))

            # 模型输出的mask， *10看着更加明显
            cv2.imwrite("model_mask.jpg", parsing * 10)

            vis_parsing_maps(image, parsing, stride=1, save_im=True,
                             save_path=osp.join(respth, image_path))


if __name__ == "__main__":
    evaluate(dspth='./image', cp='79999_iter.pth')

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

from utilities import load_saved, process_image


def model_prediction(image_path, model, topk, device):
    if torch.cuda.is_available() and device == 'gpu':
        model.to('cuda')

    img = process_image(image_path)
    img = img.unsqueeze_(0)
    img = img.float()

    if device == 'gpu':
        with torch.no_grad():
            output = model.forward(img.cuda())
    else:
        with torch.no_grad():
            output = model.forward(img)

    prob = torch.exp(output)
    top_p, top_class = prob.topk(topk)

    idx_to_class = {val: key for key, val in
                    model.class_to_idx.items()}
    mapped_classes = list()

    #      print('TOP CLASS', top_class.cpu().numpy())

    for label in top_class.cpu().numpy()[0]:
        mapped_classes.append(idx_to_class[label])

    return top_p, mapped_classes


def main():
    parsit = argparse.ArgumentParser(description='Prediction with probabilities')

    parsit.add_argument('input', default='./flowers/test/1/image_06752.jpg', nargs='?', action="store", type=str)
    parsit.add_argument('--dir', action="store", dest="data_dir", default="./flowers/")
    parsit.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type=str)
    parsit.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
    parsit.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parsit.add_argument('--gpu', default="gpu", action="store", dest="gpu")

    args = parsit.parse_args()
    image_path = args.input
    topk = args.top_k
    device = args.gpu

    chk_path = 'checkpoint.pth'
    if args.checkpoint:
        chk_path = args.checkpoint

    model = load_saved(chk_path)

    with open('cat_to_name.json', 'r') as json_file:
        cat_to_name = json.load(json_file)

    probs, flower_names = model_prediction(image_path, model, topk, device)

    name = flower_names[0]

    names = []

    for class_idx in flower_names:
        names.append(cat_to_name[class_idx])
    #     print('NAMES', names)

    probs = probs.cpu().numpy()

    # print('PROBABILITY', probs[0][1])
    print('TOP {} PREDICTIONS:'.format(topk))
    i = 0
    while i < topk:
        print("{} with a probability of {:.4f}".format(names[i], probs[0][i]))
        i += 1
    print("Predict.py satisfying rubrics")


if __name__ == "__main__":
    main()
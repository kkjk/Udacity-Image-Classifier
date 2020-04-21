import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, tensor, optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from PIL import Image


def data_preprocessing(folder):
    data_dir = folder
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # transforms for the training, validation, and testing sets

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    valid_tranforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    #  Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_tranforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    return train_data, valid_data, test_data


def data_loader(folder):
    data_dir = folder

    train_data, valid_data, test_data = data_preprocessing(data_dir)

    #  Using the image datasets and the transforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=40, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=40)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=40)

    return trainloader, validloader, testloader


def save_chkpt(model, train_data, save_dir):
    filename = 'checkpoint.pth'
    file = save_dir + filename

    model.class_to_idx = train_data.class_to_idx
    model_state = {

        'state_dict': model.state_dict(),
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
    }
    chkpt_model = torch.save(model_state, file)

    return chkpt_model, filename


def load_saved(chkfile):
    checkpt = torch.load(chkfile)

    model = models.vgg16(pretrained=True)
    model.state_dict = checkpt['state_dict']
    model.classifier = checkpt['classifier']
    model.class_to_idx = checkpt['class_to_idx']

    return model


def process_image(image_path='/flowers/test/1/image_06752.jpg'):
    flower = Image.open(image_path)

    disp_img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    flower_proc = disp_img(flower)
    return flower_proc





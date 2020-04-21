import numpy as np
import torch
import argparse
import torch.nn.functional as F
from torch import nn, tensor, optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from utilities import data_preprocessing, data_loader, save_chkpt


def model_training(folder, model_arch, l_r, hidden, epochs, device):
    trainloader, validloader, testloader = data_loader(folder)

    arch_mappings = {'vgg16': models.vgg16(pretrained=True),
                     'vgg13': models.vgg13(pretrained=True),
                     'resnet34': models.resnet34(pretrained=True)}

    if model_arch not in arch_mappings.keys():
        raise AssertionError("Choose one from vgg16, vgg13 or resnet34 as model architecture")

    model = arch_mappings[model_arch]

    layer_size = model.classifier[0].in_features

    for params in model.parameters():
        params.requires_grad = False

    classifier = nn.Sequential(nn.Linear(layer_size, hidden),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(hidden, 102),
                               nn.LogSoftmax(dim=1))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=l_r)

    if device == 'cuda':
        model.cuda()
    else:
        model.cpu()

    epochs = epochs
    steps = 0
    check_every = 40

    for n in range(epochs):
        #         print('Starting Training: 1 ')
        model_loss = 0

        for flowers, names in trainloader:

            steps += 1

            flowers, names = flowers.to(device), names.to(device)

            optimizer.zero_grad()

            logpred = model.forward(flowers)
            loss = criterion(logpred, names)
            loss.backward()
            optimizer.step()

            model_loss += loss.item()

            #             print('Training step completed: 2,3')

            if steps % check_every == 0:
                #                 print('Starting Training: 4 ')
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    #                     print('Starting Training: 5 ')
                    for flowers, names in validloader:
                        #                         print('Starting Training: 6 ')
                        flowers, names = flowers.to(device), names.to(device)
                        logpred = model(flowers)
                        batch_loss = criterion(logpred, names)

                        valid_loss = batch_loss.item()

                        # accuracy

                        prob = torch.exp(logpred)
                        top_p, top_class = prob.topk(1, dim=1)
                        equals = top_class == names.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print("Epoch: {}/{} | ".format(n + 1, epochs),
                      "Training Loss: {:.4f} | ".format(model_loss / check_every),
                      "Validation Loss: {:.4f} | ".format(valid_loss / len(testloader)),
                      "Validation Accuracy: {:.4f}".format(accuracy / len(testloader)))
                model_loss = 0
                model.train()
    print('End of Training!')
    return model


def main():
    # Command line arguments
    parsit = argparse.ArgumentParser(description='Training Image classsifier on a data set')

    parsit.add_argument('--gpu', action='store_true', \
                        help='Use GPU for inference if available')
    parsit.add_argument('data_dir', type=str, \
                        help='Path of the Dataset (with train, valid and test folders)')
    parsit.add_argument('--save_dir', type=str, \
                        help='Directory or path to save checkpoints')
    parsit.add_argument('--arch', type=str, \
                        help='Model architeture. Deafult is vgg16. Choose one from vgg16, vgg13 and resnet34')
    parsit.add_argument('--learning_rate', type=float, \
                        help='Learning rate. Default lr is  0.001')
    parsit.add_argument('--hidden_units', type=int, \
                        help='Hidden units. Default hidden units: 784')
    parsit.add_argument('--epochs', type=int, \
                        help='Number of epochs. Default is 5')

    args, _ = parsit.parse_known_args()

    folder = args.data_dir

    save_dir = './'
    if args.save_dir:
        save_dir = args.save_dir

    model_arch = 'vgg16'
    if args.arch:
        model_arch = args.arch

    l_r = 0.001
    if args.learning_rate:
        l_r = args.learning_rate

    hidden = 784
    if args.hidden_units:
        hidden = args.hidden_units

    epochs = 5
    if args.epochs:
        epochs = args.epochs

    device = 'cuda'
    if args.gpu:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    train_data, valid_data, test_data = data_preprocessing(folder)

    trainloader, validloader, testloader = data_loader(folder)

    my_model = model_training(folder, model_arch, l_r, hidden, epochs, device)

    save_chkpt(my_model, train_data, save_dir)

    print('Train.py rubrics done!')


if __name__ == "__main__":
    main()








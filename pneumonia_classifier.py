#! /usr/bin/env python3

"""
@authors: Matthew Carter   (carter36@wwu.edu)
          Raleigh Hansen   (hansen92@wwu.edu)
          Ivan Chuprinov   (chuprii@wwu.edu)
          Chris Drazic     (drazicc@wwu.edu)
          Alex Gavin       (gavina2@wwu.edu)

CNN driver for pneumonia classification.
Minimal code extended from Brian Hutchinson's CSCI597J Lab 4

For usage, run with the -h flag.

"""
import argparse
import errno
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import wandb

from operator import add
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
from torch.nn import Linear, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module
from torch.utils.data import DataLoader

import architectures.blocks as blocks
from architectures.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


class PyTorchResNet(torch.nn.Module):
    def __init__(self, c: int, C: int, v: int, pretrained: bool):
        """
            Initializes a specified torchvision.models.resnet{v} model.
        :param c: Number of channels
        :param C: Number of classes
        :param v: Version of ResNet to use
        :param pretrained: If the model should be pretrained
        """
        super(PyTorchResNet, self).__init__()

        if v == 18:
            self.model = torchvision.models.resnet18(pretrained)
        elif v == 34:
            self.model = torchvision.models.resnet34(pretrained)
        elif v == 50:
            self.model = torchvision.models.resnet50(pretrained)
        elif v == 101:
            self.model = torchvision.models.resnet101(pretrained)
        elif v == 152:
            self.model = torchvision.models.resnet152(pretrained)
        else:
            print(f"\tError: \"ResNet{v}\" is not a valid model function.")
            exit()

        self.model.conv1 = nn.Conv2d(c, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.conv = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool
        )

        self.layers = nn.Sequential(
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4
        )

        num_features = self.model.fc.in_features
        self.fc = nn.Linear(num_features, C)

    def forward(self, x):
        y_pred = self.conv(x)
        y_pred = self.layers(y_pred)

        y_pred = self.model.avgpool(y_pred)
        y_pred = torch.flatten(y_pred, 1)
        y_pred = self.fc(y_pred)

        return y_pred


class SimpleConvNeuralNetwork(Module):
    def __init__(self, C, f1):
        """
            In the constructor we instantiate convolutional, pooling, and linear layers.
            Print shapes to ensure proper dimensionality.
        :param C: Number of classes
        :param f1: Activation function {tanh, sigmoid, relu}
        """
        super(SimpleConvNeuralNetwork, self).__init__()

        if f1 == "tanh":
            self.activation = torch.nn.Tanh()
        elif f1 == "sigmoid":
            self.activation = torch.nn.Sigmoid()
        else:
            self.activation = torch.nn.ReLU()

        self.cnn_layers = Sequential(
            Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=3, padding=0),
            self.activation,
            MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.linear_layers = Sequential(
            Linear(37 * 37 * 8, 128),
            self.activation,
            Linear(128, C)
        )

        for name, param in self.named_parameters():
            print(name, param.data.shape)

    def forward(self, x):
        """
            In the forward function we accept a Tensor of input data and we must
            return a Tensor of output data. We use Modules defined in the
            constructor as well as arbitrary operators on Tensors.
        :param x: Input image
        """

        y_pred = self.cnn_layers(x)
        y_pred = y_pred.view(y_pred.size(0), -1)
        y_pred = self.linear_layers(y_pred)

        return y_pred


def parse_all_args():
    # Parses commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="{The model to be used for pneumonia classification: \"baseline\", "
                                                "\"pytorch_resnet\", or \"resnet\" (str) [default: baseline]}",
                        default="baseline")
    parser.add_argument("C", type=int,
                        help="The number of classes if classification or output dimension if regression (int) [default:"
                             " 2]", default=2)
    parser.add_argument("train", help="Root to directory containing the training subdirectories input data")
    parser.add_argument("dev", help="Root to directory containing the development subdirectories input data")
    parser.add_argument("augment_type", help='''Comma-delimited augmentation categories for data preprocessing.
                                                Available augmentation commands: brightness, contrast, rotation, hflip, vflip.
                                                The values can be specified as follows: "command_name(str)=value(float)"
                                                Example: brightness=1.5,contrast=2,rotation,vflip,hflip=rand
                                                Boosts brightness values by 1.5, doubles contras,
                                                rotates a random number of degrees out of [0,90,180,270],
                                                flips along vertical axis and randomly (50%% of the time) flips along horizontal axis
                                                ''')

    parser.add_argument("-c", type=int, help="The number of input channels (int) [default: 1]", default=1)
    parser.add_argument("-v", type=int, help="The implementation of ResNet to use: \"18\", \"34\", \"50\", \"101\","
                                             " or \"152\" (int) [default: 18]", default=18)
    parser.add_argument("-p", type=bool, help="Should ResNet use the \"pretrained\" implementation: \"True\" or "
                                              "\"False\" (bool) [default: False]", default=False)
    parser.add_argument("-lr", type=float,
                        help="The learning rate (float) [default: 0.1]", default=0.1)
    parser.add_argument("-mb", type=int,
                        help="The minibatch size (int) [default: 32]", default=32)
    parser.add_argument("-report_freq", type=int,
                        help="Dev performance is reported every report_freq updates (int) [default: 128]", default=128)
    parser.add_argument("-epochs", type=int,
                        help="The number of training epochs (int) [default: 100]", default=100)
    parser.add_argument("-f1", type=str,
                        help="The hidden activation function: \"relu\" or \"tanh\" or \"sigmoid\" (string) [default: "
                             "\"relu\"]",
                        default="relu")
    parser.add_argument("-opt", type=str,
                        help="The optimizer: \"adadelta\", \"adagrad\", \"adam\",\"rmsprop\", \"sgd\" (string) ["
                             "default: \"adam\"]",
                        default="adam")
    parser.add_argument("-workers", type=int,
                        help="The number of workers used to load the datasets (int) [default: 4]", default=4)
    parser.add_argument("-test", type=str, help="Switch to evaluate mode by including root to directory for"
                                                " test data (str) [default: \"\"]", default="")
    parser.add_argument("-model", type=str, help="Full path to trained model to be tested", default="")

    return parser.parse_args()


def choose_optimizer(opt, params, lr):
    optimizer = None

    if opt == "adadelta":
        optimizer = torch.optim.Adadelta(params, lr=lr)
    elif opt == "adagrad":
        optimizer = torch.optim.Adagrad(params, lr=lr)
    elif opt == "adam":
        optimizer = torch.optim.Adam(params, lr=lr)
    elif opt == "rmsprop":
        optimizer = torch.optim.RMSprop(params, lr=lr)
    elif opt == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr)
    else:
        print(f"\tError: \"{opt}\" is an invalid/unimplemented optimizer.")
        exit()

    return optimizer


def choose_transforms(augment_type: str):
    """
        Using command line arguments, setup data augmentation using transforms.Compose()
    :param augment_type: String specifying which specific data augmentation suite to apply.
    :return: transforms.Compose() object
    """
    transforms_list = [transforms.Grayscale()]

    for transform in augment_type.split(","):
        transform = transform.split("=")

        if transform[0] == "brightness":
            # TODO would like a way to uniformly raise the values instead of scaling them
            if len(transform) == 2:
                transforms_list.append(
                    transforms.Lambda(lambda x: transforms.functional.adjust_brightness(x, float(transform[1]))))
            else:
                # Double the brightness values
                transforms_list.append(transforms.Lambda(lambda x: transforms.functional.adjust_brightness(x, 2)))

        elif transform[0] == "contrast":
            if len(transform) == 2:
                transforms_list.append(
                    transforms.Lambda(lambda x: transforms.functional.adjust_contrast(x, float(transform[1]))))
            else:
                # Double the contrast
                transforms_list.append(transforms.Lambda(lambda x: transforms.functional.adjust_contrast(x, 2)))

        elif transform[0] == "rotation":
            if len(transform) == 2:
                # Clockwise rotation by the specified number of degrees
                transforms_list.append(transforms.Lambda(
                    lambda x: transforms.functional.affine(x, float(transform[1]), (0, 0), 1, (0, 0))))
            else:
                # Random rotations (0/90/180/270)
                transforms_list.append(RandomRotationTransform())

        elif transform[0] == "hflip":
            if len(transform) == 2 and "random".startswith(transform[1]):
                # 50% of the time, flips horizontally
                transforms_list.append(
                    transforms.RandomApply([transforms.Lambda(lambda x: transforms.functional.hflip(x))]))
            else:
                # 100% of the time, flips horizontally
                transforms_list.append(transforms.Lambda(lambda x: transforms.functional.hflip(x)))

        elif transform[0] == "vflip":
            if len(transform) == 2 and "random".startswith(transform[1]):
                # 50% of the time, flips vertically
                transforms_list.append(
                    transforms.RandomApply([transforms.Lambda(lambda x: transforms.functional.vflip(x))]))
            else:
                # 100% of the time, flips vertically
                transforms_list.append(transforms.Lambda(lambda x: transforms.functional.vflip(x)))

    transforms_list.append(transforms.Resize((224, 224)))
    transforms_list.append(transforms.ToTensor())
    return transforms.Compose(transforms_list)


class RandomRotationTransform:
    """
        Rotate by a random right angle.
    """

    def __init__(self):
        self.angles = [-90, 0, 90, 180]

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle, fill=(0,))


def choose_model(args):
    if args.model == "pytorch_resnet":
        model = PyTorchResNet(args.c, args.C, args.v, args.p)  # hardcoded num channels, using grayscale input = 1 channel
    elif args.model == "resnet":
        if args.v == 18:
            model = resnet18(args.c, args.C, block=blocks.ResNetBasicBlock)
        elif args.v == 34:
            model = resnet34(args.c, args.C, block=blocks.ResNetBasicBlock)
        elif args.v == 50:
            model = resnet50(args.c, args.C, block=blocks.ResNetBasicBlock)
        elif args.v == 101:
            model = resnet101(args.c, args.C, block=blocks.ResNetBasicBlock)
        elif args.v == 152:
            model = resnet152(args.c, args.C, block=blocks.ResNetBasicBlock)
        else:
            model = resnet18(args.c, args.C, block=blocks.ResNetBasicBlock)
    else:
        model = SimpleConvNeuralNetwork(args.C, args.f1)

    return model


def save_model(model, args, run_name: str):
    saved_models_dir = "saved_models"

    if not os.path.isdir(saved_models_dir):
        os.mkdir(saved_models_dir)
    if not os.path.isdir(saved_models_dir + "/" + args.model):
        os.mkdir(saved_models_dir + "/" + args.model)

    if run_name is None or run_name == "":
        if args.model == "pytorch_resnet" or args.model == "resnet":
            model_name = args.model + str(args.v)
        else:
            model_name = args.model
    else:
        model_name = run_name

    model_file_name = model_name + "_" + args.augment_type + ".ckpt"

    torch.save(model.state_dict(), saved_models_dir + "/" + args.model + "/" + model_file_name)


def train(model, train_loader, dev_loader, args, device):
    criterion = CrossEntropyLoss(reduction='sum')
    optimizer = choose_optimizer(args.opt, model.parameters(), args.lr)

    for epoch in range(args.epochs):

        for update, (mb_x, mb_y) in enumerate(train_loader):

            mb_y_pred = model(mb_x.to(device))  # Evaluate model forward function
            loss = criterion(mb_y_pred.to(device), mb_y.to(device))  # Compute loss

            optimizer.zero_grad()  # Reset the gradient values
            loss.backward()  # Compute the gradient values
            optimizer.step()  # Apply gradients

            if (update % args.report_freq) == 0:
                # Eval on dev once per report frequency
                acc, true, pred, _ = evaluate(model, dev_loader, device, False)
                print("%03d.%04d: dev %.3f" % (epoch, update, acc))
                print("f1 this eval: ", f1_score(true, pred, average='binary'))

    # Final evaluation
    acc, true, pred, _ = evaluate(model, dev_loader, device, False)
    f1 = f1_score(true, pred, average='binary')
    precision = precision_score(true, pred, average='binary')
    recall = recall_score(true, pred, average='binary')
    print("f1 this eval: ", f1)
    print("precision this eval: ", precision)
    print("recall this eval: ", recall)

    return f1, precision, recall, acc


def test(device, args, test_loader):
    slash_slash = args.model.split("//")[1]
    augm_type = slash_slash.split("_")[2].split(".")[0]  # Hard coded magic number :^)

    # Init wandb
    wandb.init(project="test_" + augm_type)
    wandb.init(tags=["test"])

    path = args.model

    # Load model and turn off training mode
    if "baseline" in path:
        model = SimpleConvNeuralNetwork(args.C, args.f1).to(device)
    elif "pytorch_resnet" in path:
        model = PyTorchResNet(args.c, args.C, args.v, args.p).to(device)
    else:
        print("Unknown model type or misnamed directory in path:", path)
        exit()

    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model.eval()

    # Perform evaluation and log
    acc, true, pred, pred_probs = evaluate(model, test_loader, device, True)

    # Get name of model from path
    path_pieces = path.split('/')
    state_dict = [i for i in path_pieces if '.ckpt' in i]

    f1 = f1_score(true, pred, average='binary')
    wandb.log({
        "model": state_dict[0],
        "f1": f1,
        "accuracy": acc,
        'pr': wandb.plots.precision_recall(true, pred_probs, ["normal", "pneumonia"])
    })

    wandb.sklearn.plot_confusion_matrix(true, pred, ["normal", "pneumonia"])

    save_eval(true, pred, pred_probs, state_dict[0].split('.ckpt'))

    print("f1 for model ", str(state_dict[0] + ": "), f1)


def evaluate(model, loader, device, is_test):
    acc_num = 0
    acc_denom = 0

    cumm_true = torch.empty(0, dtype=torch.long).to(device)
    cumm_pred = torch.empty(0, dtype=torch.long).to(device)

    # If testing (done on cpu) keep track of probabilities
    if is_test:
        cumm_pred_probs = torch.empty(0, dtype=torch.float32).to(device)
    else:
        cumm_pred_probs = 0

    # Gather model predictions and ground truth
    for update, (x, y) in enumerate(loader):
        y_pred = model(x.to(device))

        _, y_pred_i = torch.max(y_pred.to(device), 1)
        y_pred_i = y_pred_i.to(device)
        y = y.to(device)

        acc_num += (y_pred_i == y).sum().item()
        acc_denom += loader.batch_size

        cumm_true = torch.cat((cumm_true, y))
        cumm_pred = torch.cat((cumm_pred, y_pred_i))

        # If testing, calculate and store prediction probs
        if is_test:
            probs = nn.functional.softmax(y_pred.float(), dim=1)
            cumm_pred_probs = torch.cat((cumm_pred_probs, probs))

    if acc_denom != 0:
        acc = acc_num / acc_denom
    else:
        acc = 0

    # sklearn computation uses ndarrays and thus must be on cpu
    cumm_true = cumm_true.to("cpu")
    cumm_pred = cumm_pred.to("cpu")

    if is_test:
        cumm_pred_probs = cumm_pred_probs.to("cpu")
        cumm_pred_probs = cumm_pred_probs.detach().numpy()

    return acc, cumm_true, cumm_pred, cumm_pred_probs


def save_eval(true, pred, probs, model):
    # Compose path string for specific augmentation type
    pieces = model[0].split('_')
    arch_aug = pieces[0] + '_' + pieces[1] + '_' + pieces[2]
    new_path = 'saved_evals/' + arch_aug + "/"

    if len(pieces) < 4:
        pieces.append("")

    test_path = new_path + "true_" + pieces[3]
    pred_path = new_path + "pred_" + pieces[3]
    prob_path = new_path + "prob_" + pieces[3]

    # Create paths if they do not exist
    paths = [test_path, pred_path, prob_path]
    for path in paths:
        if not os.path.exists(os.path.dirname(path)):
            try:
                os.makedirs(os.path.dirname(test_path))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    np.save(test_path, true)
    np.save(pred_path, pred)
    np.save(prob_path, probs)


def average_scores(path):
    f_pred = True
    f_prob = True
    count = 0
    pred = []
    prob = []
    true = []
    all_pred = []
    all_prob = []

    # Add predictions and probabilities for specific augmentation
    for root, dirs, files in os.walk(path):
        for file in files:
                
            if "pred" in file:
                pred = np.load(root + file)
                count += np.count_nonzero(pred, axis=0)

                if f_pred:
                    all_pred = [0]*len(pred)
                    f_pred = False

            elif "prob" in file:
                prob = np.load(root + file)
                prob.sort()
                if f_prob:
                    all_prob = [0]*len(prob)
                    f_prob = False

                all_prob = list(map(add, prob, all_prob))
            elif "true" in file:
                true = np.load(root + file)
                true.sort()

    root = root.split('/')

    # Average over number of trained models
    denom = len(files)/3
    count = count/denom
    avg_pred = np.ones(int(count))
    for i in range(len(pred)-len(avg_pred)):
        avg_pred = np.append(avg_pred, [0])
    prob = np.divide(all_prob, denom)
    
    # Init wandb
    wandb.init(project="final_final_averaging_")
    wandb.init(tags=["test"])

    wandb.log({
        "model": root[1],
        "f1": f1_score(true, pred, average='binary')
    })
    wandb.sklearn.plot_confusion_matrix(true, pred, ["normal", "pneumonia"])


def main():
    # Parse arguments
    args = parse_all_args()
    img_transform = choose_transforms(args.augment_type)

    # If in testing mode, perform tests on cpu and exit
    if args.test != "":
        device = torch.device("cpu")

        if "evals" in args.model:
            average_scores(args.model)
            exit()

        # Recommended not to perform augmentations on test data.
        test_data = torchvision.datasets.ImageFolder(root=args.test, transform=img_transform)
        test_loader = DataLoader(test_data, batch_size=args.mb, shuffle=True, drop_last=False, num_workers=args.workers)

        test(device, args, test_loader)
        exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init train and test dataloaders using specified transform(s)
    train_data = torchvision.datasets.ImageFolder(root=args.train, transform=img_transform)
    dev_data = torchvision.datasets.ImageFolder(root=args.dev, transform=img_transform)

    train_loader = DataLoader(train_data, batch_size=args.mb, shuffle=True, drop_last=False, num_workers=args.workers)
    dev_loader = DataLoader(dev_data, batch_size=args.mb, shuffle=False, drop_last=False, num_workers=args.workers)

    model = choose_model(args).to(device)

    # Init wandb
    wandb.init(entity="597J_project")
    wandb.init(project="pneumonia_classifier_" + args.model)
    wandb.init(tags=[args.augment_type])
    wandb.config.update(args)

    f1, precision, recall, dev_acc = train(model, train_loader, dev_loader, args, device)

    print("--- Final evaluation scores --- ")
    print(f"f1: {f1}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")

    # Save model
    save_model(model, args, wandb.run.name)

    wandb.log({
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": dev_acc
    })


if __name__ == "__main__":
    main()

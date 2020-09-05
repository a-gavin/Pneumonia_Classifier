#! /usr/bin/env python3

"""
@authors: Matthew Carter   (carter36@wwu.edu)
          Raleigh Hansen   (hansen92@wwu.edu)
          Ivan Chuprinov   (chuprii@wwu.edu)
          Chris Drazic     (drazicc@wwu.edu)
          Alex Gavin       (gavina2@wwu.edu)

Script to train pytorch_resnet using different data augmentations.

"""
import subprocess

# Arguments used to spawn subprocess, must separate on whitespace
args = ["python3", "pneumonia_classifier.py", 
        "pytorch_resnet", 
        "2", 
        "../pneumonia_classifier/chest_xray/train", "../pneumonia_classifier/chest_xray/val",
        "default", 
        "-v", "34",
        "-mb", "8", 
        "-workers", "8", 
        "-e", "25", 
        "-opt", "adam", 
        "-lr", "0.001"]

transforms = ["default",
              "brightness",
              "contrast",
              "rotation",
              "brightness,contrast",
              "brightness,rotation",
              "contrast,rotation"]

for transform in transforms:
    args[6] = transform

    for i in range(5):
        subprocess.check_call(args)

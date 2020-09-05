#! /usr/bin/env python3

"""
@authors: Matthew Carter   (carter36@wwu.edu)
          Raleigh Hansen   (hansen92@wwu.edu)
          Ivan Chuprinov   (chuprii@wwu.edu)
          Chris Drazic     (drazicc@wwu.edu)
          Alex Gavin       (gavina2@wwu.edu)

Script to test pre-trained models located in saved_models.
Results stored in saved_evals.

"""
import subprocess
import os

# Arguments used to spawn subprocess, must separate on whitespace
# -model argument set in main for loop
args = ["python3", "pneumonia_classifier.py", 
        "baseline", 
        "2", 
        "chest-xray-pneumonia/chest_xray/train/", "chest-xray-pneumonia/chest_xray/test/",
        "default", 
        "-mb", "16", 
        "-workers", "8", 
        "-e", "5", 
        "-opt", "adam", 
        "-test", "chest-xray-pneumonia/chest_xray/val/",
        "-model"]

# Iterate over directories for all architectures
for root, dirs, architectures in os.walk("saved_models/"):

    # Iterate over models for all augmentation types
    for state_dict in architectures:
        path = str(root + '/' + state_dict)
        args.append(path)
        subprocess.check_call(args)
        args.pop()

for root, dirs, aug in os.walk("saved_evals/"):

    # Iterate over models for all augmentation types
    for d in dirs:
        path = str(root + d + '/')
        args.append(path)
        subprocess.check_call(args)
        args.pop()

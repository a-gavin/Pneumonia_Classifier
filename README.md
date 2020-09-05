# Chest X-Ray Pneumonia Classifier

### Description
Collaborative project for CSCI597J (Deep Learning, now listed as [CSCI581][1]) to determine the relative effect of data augmentations in classifying pneumonia from chest X-rays.

###### Tools used: PyTorch, WandB, scikit-learn Metrics, HTCondor

For this task, we chose a widely available [chest X-ray data set][2] consisting of 5,863 labeled chest X-rays. 
To load and perform augmentations on this data, our classifier uses the torchvision package's 
[ImageFolder][3] class and [Transforms][4] subpackage. Using Pytorch's [DataLoader][5] class, this data is fed to one
of several CNNs, including several implementations of the popular [ResNet][6] family of CNNs.

Since the data is imbalanced, we use [scikit-learn's implementations][7] of precision, recall, and F1 score
functions to evaluate trained models with as little class bias as possible. Coupled with [WandB][8] for logging
and [HTCondor][9] on the WWU compute cluster, the program is able to concurrently train and visualize experiment results in real-time.

For a detailed analysis of our experiments and results, please see the [project report][10].

### Project Structure
1. **[pneumonia_classifier.py][11]**: 

    Driver program to train/test models. Option to use baseline CNN or a selection of ResNet architectures, including PyTorch's.
    See [Usage](#usage) for example usage.

2. **[train_script.py][12]**, **[test_script.py][13]**: 

    Scripts to perform training and testing, respectively, using pre-selected data augmentations 
    (see Section IV of our [final report][10]).
    
3. **deliverables**:

    Submissions for project assignments. Includes status reports, proposals, [final report][10], and [presentation slides][14].
    
4. **architectures**:
    
    Implementations of ResNet18, ResNet34, ResNet50, ResNet101, and ResNet152
    
### Usage
Below is the most basic usage of the **[pneumonia_classifier.py][11]** program. More details about arguments are available by invoking the program with the **-h** flag.
    
    python3 pneumonia_classifier.py <architecture> <classes> <path_to_train_data> <path_to_val_data> <augmentation_type>
    
Example Invocations:

    # Increase brightness by 2.5x using baseline CNN
    python3 pneumonia_classifier.py baseline 2 ~/path/to/train/data ~/path/to/val/data brightness=2.5 -e 20 -mb 128
    
    # No data augmentation using ResNet34
    python3 pneumonia_classifier.py resnet 2 ~/path/to/train/data ~/path/to/val/data default -v 34 -opt adam -lr 0.01
    
    # Run testing on. Train dataset required for argparse, but only test dataset is loaded.
    # TEST_NAME is name used for WandB logging. MODEL is model used to test in eval mode.
    python3 pneumonia_classifier.py resnet 2 ~/path/to/train/data ~/path/to/test/data default -test TEST_NAME -model MODEL
    
### Authors
  * Matthew Carter   (carter36 at wwu.edu)
  * Raleigh Hansen   (hansen92 at wwu.edu)
  * Ivan Chuprinov   (chuprii at wwu.edu)
  * Chris Drazic     (drazicc at wwu.edu)
  * Alex Gavin       (gavina2 at wwu.edu)

[1]: https://catalog.wwu.edu/preview_course_nopop.php?catoid=16&coid=125176& "CSCI581 Course Description"
[2]: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia "Kaggle Chest X-Ray Data Set"
[3]: https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder "ImageFolder Documentation"
[4]: https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision-transforms "Transforms Documentation"
[5]: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader "DataLoader Documentation"
[6]: https://arxiv.org/pdf/1512.03385.pdf "ResNet Paper"
[7]: https://scikit-learn.org/stable/modules/model_evaluation.html "scikit-learn Metrics Documentation"
[8]: https://www.wandb.com/ "WandB Website"
[9]: https://research.cs.wisc.edu/htcondor/ "HTCondor Website"
[10]: https://github.com/agavin97/pneumonia_classifier/blob/master/deliverables/report.pdf "Final Report"
[11]: https://github.com/agavin97/pneumonia_classifier/blob/master/pneumonia_classifier.py "pneumonia_classifier.py"
[12]: https://github.com/agavin97/pneumonia_classifier/blob/master/train_script.py "train_script.py"
[13]: https://github.com/agavin97/pneumonia_classifier/blob/master/test_script.py "test_script.py"
[14]: https://github.com/agavin97/pneumonia_classifier/blob/master/deliverables/slides.pdf "Presentation Slides"


# X-Ray image classification

Dataset: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia.  
You can download data using: `python -m download`

This is a simple baseline binary classifier to predict if the image
shows pneumonia or not.
Baseline model is a pretrained `resnet34`.

## Training the model

`python -m train`

Parameters:  
`--lr <float>`: learning rate  
`-e <int>`: epochs  
`--log_dir <string>`: logs directory

Visualize learning progress using Tensorboard.

## Viral vs Bacterial Penumonia

This model can also learn to classify types of Pneumonia, by using only
the images from the set that belong to pneumonia class. Annotations for bacterial vs pneumonia are provided in image file name, and the `XRayVBDataset` will provide them.

Train model for viral vs bacterial pneumonia:  
`python -m train -d vb`

## Pending

Create custom model and perform multi-class classification:  
classes: [normal, viral, bacterial], using cross-entropy loss.

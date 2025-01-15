# Project Name  Melanoma Skin Cancer Detection
> Outline a brief description of your project.
Problem statement: To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis

## Table of Contents
* [General Info](#general-information)
The dataset comprises 2357 images depicting malignant and benign oncological conditions, sourced from the International Skin Imaging Collaboration (ISIC). These images were categorized based on the classification provided by ISIC, with each subset containing an equal number of images.
Model Architecture
The break down of the final provided CNN architecture step by step:

Data Augmentation: The augmentation data variable refers to the augmentation techniques applied to the training data. Data augmentation is used to artificially increase the diversity of the training dataset by applying random transformations such as rotation, scaling, and flipping to the images. This helps in improving the generalization capability of the model.

Normalization: The Rescaling(1./255) layer is added to normalize the pixel values of the input images. Normalization typically involves scaling the pixel values to a range between 0 and 1, which helps in stabilizing the training process and speeding up convergence.

Convolutional Layers: Three convolutional layers are added sequentially using the Conv2D function. Each convolutional layer is followed by a rectified linear unit (ReLU) activation function, which introduces non-linearity into the model. The padding='same' argument ensures that the spatial dimensions of the feature maps remain the same after convolution. The number within each Conv2D layer (16, 32, 64) represents the number of filters or kernels used in each layer, determining the depth of the feature maps.

Pooling Layers: After each convolutional layer, a max-pooling layer (MaxPooling2D) is added to downsample the feature maps, reducing their spatial dimensions while retaining the most important information. Max-pooling helps in reducing computational complexity and controlling overfitting.

Dropout Layer: A dropout layer (Dropout) with a dropout rate of 0.2 is added after the last max-pooling layer. Dropout is a regularization technique used to prevent overfitting by randomly dropping a fraction of the neurons during training.

Flatten Layer: The Flatten layer is added to flatten the 2D feature maps into a 1D vector, preparing the data for input into the fully connected layers.

Fully Connected Layers: Two fully connected (dense) layers (Dense) are added with ReLU activation functions. The first dense layer consists of 128 neurons, and the second dense layer outputs the final classification probabilities for each class label.

Output Layer: The number of neurons in the output layer is determined by the target_labels variable, representing the number of classes in the classification task. The output layer does not have an activation function specified, as it is followed by the loss function during training.

Model Compilation: The model is compiled using the Adam optimizer (optimizer='adam') and the Sparse Categorical Crossentropy loss function (loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)), which is suitable for multi-class classification problems. Additionally, accuracy is chosen as the evaluation metric (metrics=['accuracy']).

Training: The model is trained using the fit method with the specified number of epochs (epochs=50). The ModelCheckpoint and EarlyStopping callbacks are employed to monitor the validation accuracy during training. The ModelCheckpoint callback saves the model with the best validation accuracy, while the EarlyStopping callback stops training if the validation accuracy does not improve for a specified number of epochs (patience=5 in this case). These callbacks help prevent overfitting and ensure that the model converges to the best possible solution.
* [Technologies Used](#technologies-used)
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
* [Conclusions](#conclusions)

* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
- Provide general information about your project here.
- What is the background of your project?
- What is the business probem that your project is trying to solve?
- What is the dataset that is being used?

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Conclusions
- Conclusion 1 from the analysis
Analyze your results here. Did you get rid of underfitting/overfitting? Did class rebalance help? Observations:

The ultimate model showcases well-balanced performance, displaying no signs of underfitting or overfitting.

The implementation of class rebalancing has notably enhanced the model's performance across both training and validation datasets.

Following 37 epochs, the final model attains an accuracy of 84% on the training set and approximately 79% on the validation set.

The narrow divergence between training and validation accuracies underscores the robust generalization capability of the final CNN model.

The addition of batch normalization failed to enhance both training and validation accuracy.

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Acknowledgements
Give credit here.
- This project was inspired by...
- References if any...
- This project was based on [this tutorial](https://www.example.com).


## Contact
Created by [@githubusername] - feel free to contact me!
https://github.com/sureshkns100/Melanoma-Detection-Assignment

<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->

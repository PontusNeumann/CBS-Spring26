

## Compulsory Assignment #3
Machine Learning and Deep Learning (CDSCO2041C)
## Somnath Mazumdar
sma.digi@cbs.dk
Department of Digitalization, Copenhagen Business School
Deadline: Check Canvas LMS
## Instructions
- Write your group members’ student IDs in the notebook.
- All explanations must be included as markdown/text cells within the notebook.
- The complete solution must be submitted asone singleJupyternotebook.
Assignment: Perform image classification
You must use theCIFAR-100 dataset, which contains 60,000 colour images of size32×32pixels across
100 fine-grained classes. The dataset provides 50,000 training images and 10,000 test images. Useone
shared preprocessing pipelineacross all parts. Download the data from here.
## 1.1. Data Preparation
Note:The flattened feature vectors produced in this section are used for pixel-based classification (Part 1.3).
The original spatial image arrays (32×32×3) are retained separately as input to the convolutional au-
toencoder in Part 1.2.
•Load the CIFAR-100 train and test splits
•Create a validation split from the training set using a 90/10 ratio
•Compute theper-channel mean and standard deviationusing the training set only
•Normalize the training, validation, and test images using these statistics
•Flatten each normalized image into a feature vector of size32×32×3 = 3072
1.2. Autoencoder-Based Feature Learning
Construct a convolutional autoencoder and train it on the normalized training images (using the spatial
32×32×3format,notthe flattened vectors).
•Use an encoder-decoder architecture with at leasttwo convolutional blocksin the encoder and a
symmetric decoder
•Train the model usingreconstruction loss(e.g., MSE)
•Use the trained encoder to extract latent feature vectors for the train, validation, and test sets
## 1

1.3. Classification Using Pixel and Latent Features
Train a simple classifier (e.g., a linear classifier ork-nearest neighbours) on each of the two feature sets
to evaluate their relative utility.
•Train and evaluate one classifier on theflattened pixel featuresfrom Part 1.1
•Train and evaluate one classifier on thelatent featuresextracted by the encoder in Part 1.2
•Use thesame classifier type and hyperparametersfor both to ensure a fair comparison
1.4. CNN Model
Implement a convolutional neural network for CIFAR-100 classification using the normalized images (spa-
tial32×32×3format).
•Useat least threeConv2Dlayers organised into two or more convolutional blocks
•ApplyReLUactivations and max pooling after each block
•Useat least onefully connected (dense) layer before the final classification layer
•The output layer must predict all100 classesusing a softmax activation
•Print the number of trainable parameters in the CNN
## 2
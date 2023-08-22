# Dataset
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

# Model Architecture
Inspired by the outstanding performance of transfer learning and ResNet, our model consists of a transfer
learning at the beginning to generate feature embeddings, and customized CNN layers similar to the
ResNet structure.

First, we use the pre-trained AlexNet on the preprocessed data (224*224 pixels) to generate feature
embeddings with dimensions 256*6*6 for each image. Then, the data are batched and fed to the CNN
layers. The CNN layers contain four ResBlocks. Inside each ResBlock, there are three layers: a
one-by-one convolution to reduce the input channels from 256 to 128, a three-by-three convolution to
extract the spatial information while remain the 128 channels, and a one-by-one convolution at the end to
restore the number of channels to 256. The three-layer architecture allows the model to have very few
training parameters and achieve the functionality of large convolution layers. Since our model
is relatively deep, we added a residual connection from the input of each block to the output of the block to
avoid the gradient vanishing or exploding. Batch normalization is also used to speed up the convergence and
improve the accuracy. Finally, there is one fully connected layer at the end to convert 256*6*6 to 3*1 for
making the prediction



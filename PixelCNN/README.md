# PixelCNN

This repository is a PyTorch implementation of [PixelCNN](https://arxiv.org/abs/1601.06759) in its [gated](https://arxiv.org/abs/1606.05328) form.
The main goals I've pursued while doing it is to dive deeper into PyTorch and the network's architecture itself, which I've found both interesting and challenging to grasp. The repo might help someone, too!

[this](http://www.scottreed.info/files/iclr2017.pdf)

# Model architecture
Here I am going to sum up the main idea behind the architecture. I won't go deep into implementation details and how convolutions work, because it would be too much text and visuals. Visit the links above in order to have a more detailed look on the inner workings of the architecture. Then come here for a summary :)

At first this architecture was an attempt to speed up the learning process of a RNN implementation of the same idea, which is a generative model that learns an explicit joint distribution of image's pixels by modeling it using simple chain rule:

<p align="center">
  <img width="353" height="54" src="http://latex.codecogs.com/gif.latex?p%28%5Cmathbf%7Bx%7D%29%20%3D%20%5Cprod_%7Bi%3D1%7D%5E%7BD%7D%20p%28x_i%5Cvert%20x_1%2C%20%5Cdots%2C%20x_%7Bi-1%7D%29%20%3D%20%5Cprod_%7Bi%3D1%7D%5E%7BD%7D%20p%28x_i%5Cvert%20x_%7B1%3Ai-1%7D%29">
</p>

The order is row-wise i.e. value of each pixel depends on values of all pixels above and to the left of it. Here is an explanatory image:

<p align="center">
  <img width="239" height="247" src="http://sergeiturukin.com/assets/2017-02-22-183010_479x494_scrot.png">
</p>
In order to achieve this property authors of the papers used simple masked convolutions, which in the case of 1-channel black and white images look like this:

<p align="center">
  <img width="403" height="256" src="https://lilianweng.github.io/lil-log/assets/images/pixel-cnn.png">
</p>
(i. e. convolutional filters are multiplied by this mask before being applied to images)


There are 2 types of masks: A and B. Masked convolution of type A can only see previously generated pixels, while mask of type B allows taking value of a pixel being predicted into consideration. Applying B-masked convolution after A-masked one preserves the causality, work it out! In the case of 3 data channels, types of masks are depicted on this image:

<p align="center">
  <img width="273" height="182" src="http://sergeiturukin.com/assets/2017-02-23-195558_546x364_scrot.png">
</p>


The problem with a simple masking approach was the blind spot: when predicting some pixels, a portion of the image did not influence the prediction. This was fixed by introducing 2 separate convolutions: horizontal and vertical.  Vertical convolution performs a simple unmasked convolution and sends its outputs to a horizontal convolution, which performs a masked 1-by-N convolution. They also added conditioning on labels and gates in order to increase the predicting power of the model.

## Gated block
The main submodel of PixelCNN is a gated block, several of which are used in the network. Here is how it looks:

![Gated block](https://github.com/anordertoreclaim/PixelCNN/blob/master/.images/gated_block.png?raw=true)

## High level architecture
Here is what the whole architecture looks like:

![PixelCNN architecture](https://github.com/anordertoreclaim/PixelCNN/blob/master/.images/architecture.png?raw=true)

Causal block is the same as gated block, except that it has neither residual nor skip connections, its input is image instead of a tensor with depth of *hidden_fmaps*, it uses mask of type A instead of B of a usual gated block and it doesn't incorporate label bias.

Skip results are summed and ran through a ReLu – 1x1 Conv – ReLu block. Then the final convolutional layer is applied, which outputs a tensor that represents unnormalized probabilities of each color level for each color channel of each pixel in the image.


# 100daysML

## 04.09.2019
### Understanding the difficulty of training deep feedforward neural networks
- https://arxiv.org/pdf/1502.01852.pdf
- https://pouannes.github.io/blog/initialization/
- http://cs231n.github.io/convolutional-networks/
- https://medium.com/impactai/cnns-from-different-viewpoints-fab7f52d159c
### Thought on paper
- the paper deals with the combination of CNN with RELU case, it shall be ok with non-CNN but with RELU
- with Kaiming intialization, it preserves either the variance of activation or back-propagation gradients, but not both. One of them will be scaled to c2/dL or dL/c2.
- it is enlighting the convolution can be simplified as a Matrix mulitplication W*x, with W containing some regularity see https://medium.com/impactai/cnns-from-different-viewpoints-fab7f52d159c

## 03.09.2019
### Understanding the difficulty of training deep feedforward neural networks
- http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
- https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/ (very comprehensive explanation)
### Thought on paper
- The initialization of weights strongly influences or determines the training process.
- standard uniform random initialization [-1/sqrt(n), 1/sqrt(n)] causes saturation of weights in deep layer network
- sigmoid activation with uniform intializatio tends to saturated neural activation at 0.5, thus causes network slow to learn
- keep the variance of activation, back gradient propagation stable, i.e. constant throughout layers,
- thus, xavier intialization of weights sqrt(6/(n1+n2)) is compromise of make variance of activation, gradient stable cross layers.

## 02.09.2019
### A Discussion of Adversarial Examples Are Not Bugs, They Are Features
- https://distill.pub/2019/advex-bugs-discussion/
### Thought on paper
- The idea is Neural network catches human-insensitive features and uses it for classification
- The human-insentive featues are not "Robust", meaning that they are random and has weak correlation to result. Does it mean that dataset does not have enough data covers the distribution of this feature, like mentioned high frequency features in image? In other words, a man with a king-crown is classified as King, here king crown is not "Robust" feature.

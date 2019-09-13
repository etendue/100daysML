# 100daysML
## 13.09.2019
### Softmax and the class of "background"
- Jerome Howard explains that Softmax needs to be used with caution, as it always convert logits to probablity, the input needs to be currated that at least one class exists, else it makes no sense when test inputs do not have any object belongs to classes; for superviosed  training it is not a problem, because the input is always labelled with a class.
- for semantic segmentation, the "background" class, i.e. none of classes wanted is hard to classify, since it needs O( N x I(class)) capacity to identify. The weights need to classify all possible classes to negative. I assume the "background" class learns the average bias or threshold of all possible classes to be classified as negative(Not a class). So it depends highly on the training distribution and has bad generalization.

## 11.09.2019
### Self-Normalizing Neural Networks
- https://arxiv.org/abs/1706.02515

### Thought on paper
- The article introduced a SELU activation function, with additional parameter of lambda and alpha
- the SELU has property to maintain the mean and variance of activation across multiple layers
- it does not consider the gradient propagation, so I am not sure if gradient vanishing or exploding problem is tackled 
- the mathematical derivation is overwhelming and initimdating. it uses ~ 100 pages to illustrate that.

### All you need is a good init
- https://arxiv.org/abs/1511.06422

### Thought on paper
- apply the weight initialization with orthogonal matrix plus scaling with batch input data
- extension from Xavier, Kaiming's initialization

### Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
- https://arxiv.org/abs/1312.6120

### Thought on paper
- apply the weight initialization with orthogonal matrix plus scaling
- extension from Xavier, Kaiming's initialization


## 09.09.2019
### A Probabilistic Representation of Deep Learning
- https://arxiv.org/abs/1908.09772

### Thought on paper
- the article tries to assemble DNNs with Graph Probablistic Model. 
- intuitively sounds a attractive idea, that layers near data learn the prior distribution of data and layers to output assemble the posterier of target information.
- Did not go through the details, so not sure if the statement is rectified in theory

## 05.09.2019
### Fixup Initialization: Residual Learning Without Normalization
- https://arxiv.org/abs/1901.09321

### Thought on paper
- the He initialization solves the Relu activation, but ResNet introduces skip-connection, which introduces new factor for gradient explosion.
- a new method is introduced to fix this new achitecture.
- for me it seems, it is always required to analyze the propoerty of data(input), the model achitecture, when an application is tried to be sovled by neural network. 
  - to check whether the solution will work at first place from theory
  - monitor the training/learning process 

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

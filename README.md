# 100daysML
## 03.10.2019
### Perceptual Losses for Real-Time Style Transfer and Super-Resolution
- https://arxiv.org/abs/1603.08155
### Thought on paper
- add feature loss function to Loss, feature loss is extracted from a transfer model
- need example and code to see how it is done

## 02.10.2019
### Visualizing the Loss Landscape of Neural Nets
- https://arxiv.org/abs/1712.09913
### Thought on paper
- non-convex loss function, why and how to know? check with Hessian matrix and its eigen value, another topic needs to learn
- how is loss surface calculated and visualized. PCA?
- is it universal by add skip-connection to convexize the loss function?
- wider net has much convexer loss surface
- The paper provides code.
- is there other tricks to convexize and smooth the loss function?


## 01.10.2019
### Deep Residual Learning for Image Recognition
- https://arxiv.org/abs/1512.03385
### Thought on paper
- I was confused why it was called residual network. The residual network here is represented by F(x), the path without skip connection.
- why it is called "residual", the difference between target output H(x) and input X. 
- It seems residual is applied for residual block, because final output of network has quite different shape as input X.
- that also bothers me when input output dimension does not match.

## 26.09.2019
### Visualizing and Understanding Convolutional Networks
- https://arxiv.org/abs/1311.2901
### Thought on paper
- Revisit the topic of feature visualization
- visualize single activation in final layer shows the decomposition from layer 1 to last
- i.e. if you visualize the last layer, you will see gradually abstracter activation maps
- if you visualize activation in ealier layer, you see the element the filter recoginize

## 24.09.2019
### A systematic study of the class imbalance problem in convolutional neural networks
- https://arxiv.org/abs/1710.05381
### Thought on paper
- several methods are experimented, oversampling, undersampling; threshold
- ROC AUC metric to measure the model. ROC AUS is another topic to explore.
- class imbalance was the first problem when I trained the CNN network with data from simulation
- I am still boggling how to handle this problem in RL domain, since in RL environment input distribution shifts during training


## 23.09.2019
### Cyclical Learning Rates for Training Neural Networks
- https://arxiv.org/pdf/1506.01186.pdf
### Thought on paper
- how to find a range of Learning rate based on trend of accurracy: when accurracy decreases while learning rate increases, set the max boundary of learning rate
- use cyclical learning rate schedule to train network instead of fixed learning rate or monotonic descreaing schedule.
- the justification of why CLR works is not very clear for me. Some hints like to add more variance of gradients?

### A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay
- https://arxiv.org/abs/1803.09820
### Thought on paper
- experiments on learning rate, batch-size, momentum and weight decay.
- introduce one-shot fit
- experiments on cyclic learning rate, moment.
- experiments on different datasets and neural network architectures
- all problems are supervised learning
- Interesting to see similar analysis in RL domain.


## 20.09.2019
### Revisiting Small Batch Training for Deep Neural Networks
- https://arxiv.org/abs/1804.07612
### Thought on paper
- Batch size affects the variance of weight update
- Big batch size with mean loss reduce the variance and degrade the generalization of SGD
- Batch Normalization counter-interact the mean effect and prefers relative large batch size.


## 19.09.2019
### Group Normalization
- https://arxiv.org/abs/1803.08494
### Thought on paper
- Generalizes the Layer Normalization and Instance Normalization
- A way to add human prior to model achitecture. I.e. we manually group the feature channel into similar catelogs.


## 17.09.2019
### Instance Normalization: The Missing Ingredient for Fast Stylization
- https://arxiv.org/abs/1607.08022
### Thought on paper
- the method is applied for Style transfer
- instance normalization as its name imply, normalization on a single image. I.e. contrast normalization


## 13.09.2019
### Softmax and the class of "background"
- Jerome Howard explains that Softmax needs to be used with caution, as it always convert logits to probablity, the input needs to be currated that at least one class exists, else it makes no sense when test inputs do not have any object belongs to classes; for superviosed  training it is not a problem, because the input is always labelled with a class.
- for semantic segmentation, the "background" class, i.e. none of classes wanted is hard to classify, since it needs O( N x I(class)) capacity to identify. The weights need to classify all possible classes to negative. I assume the "background" class learns the average bias or threshold of all possible classes to be classified as negative(Not a class). So it depends highly on the training distribution and has bad generalization.

### Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
- https://arxiv.org/pdf/1502.03167.pdf

### Thought on paper
- Batch Normalization transform the inputs of each layer to 0-mean,1 variant distribution, how would it affect the bias?
- Needs to work with a concrete example.

### Layer Normalization
- https://arxiv.org/pdf/1607.06450.pdf

### Thought on paper
- Layer normalization deals with problem of RNNs, which Batch Normalization can not
- the paper provides more insight of mathmatical properties of normalization operation, especially on invariants
- Layer normalization does not proove to be better for CNNs, as CNNs has small "perceptral angle" than Dense Network, so correlate the neuros with unrelated information is a factor that Layer Normalization may not work well
- The paper gives thinking on circumstance of problem when applying what kind of normalization. 


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

# 100daysML
## 03.03.2020
### Stacked Capsule Autoencoders
- https://arxiv.org/pdf/1906.06818.pdf
### Thought on paper
- After visiting several presentation on this topic I read the paper finally. Though still skipped some details
- Though the main idea is to add inductive prior (i.e. define affine transformation, prediction probablity and occlusion/alpha), 
- the model is realized and improved by applying several tricks. E.g. sparcity. Probabily there is model parts, CNN, autoencode have certain weakness to implement the SCAE
- it focuses mainly in vision problem, is not a generic idea. 
- As in previous paper, add known knowledge to model seems the new trend in this area.

## 29.02.2020
### Growing Neural Cellular Automata
- https://distill.pub/2020/growing-ca/
### Thought on paper
- this paper is most interesting article in recent time
- it connects my personal adventure into deep learning, complex system, e.g. CA and life science.
- in pure deep learning, many model respect(inspired by life science, e.g CNN) but not fully consider the physical world's laws. I think it is too much abstract. Similar like multi universe model vs our world in theoretical physics
- by adding bias, here the biological knowledge to neural network and computation model(cell), the result seems to approach reality.
- another article of CapsuleNetwork 2019 from Hinton seems emphasize adding known real world laws into deep learning model.


## 15.02.2020
### Visualizing the Impact of Feature Attribution Baselines
- https://distill.pub/2020/attribution-baselines/
### Thought on paper
- this paper extends the topic from last paper and focus on effect of baseline and how to choose a baseline and its pitfall
- instead of constant value baseline, several different approach is introduced, some of them are promising, e.g. distribution of data
- claim that intepretation of model shall follow human logic is not valid. Not to say we don't know exactly human intution works.e
- A question is answered with more questions but still insightful


## 09.02.2020
### Axiomatic Attribution for Deep Networks
- https://arxiv.org/pdf/1703.01365.pdf
### Thought on paper
- this is a paper to reason the deep network in sense which input (I) contributes the output(O)
- it uses a method "Integrated Gradient" which literally integrats of gradient of output/input.
- But why not gradient? the paper explains that pure gradient may lose sensitivity due to network implementation for some input
- Then how to use it? Using a network, do an inference. Then compute the "Integrated Gradient" by using Riemann integral approximation method, namely sampling n discrete values and sum/avarageing.
- The last point, I am happy that I got so much information from this paper:) 


## 24.12.2019
### Representation Learning with Contrastive Predictive Coding
- https://arxiv.org/pdf/1807.03748.pdf
### Thought on paper
- this article introduce the article of CPC and explains the motivation of extract mutual information of input and context in low dimension
- the point is how to derive and convert the input/ouput into the domain of information
- I did not get this transformation intuitively from the equation.
- when dealing with distribution, it is inevitable to confront intractable variables. Aproximate approches are needed.


## 23.12.2019
### A Simple Baseline for Bayesian Uncertainty in Deep Learning
- https://arxiv.org/abs/1902.02476
### Thought on paper
- Honestly I hardly understand the content
- It seems to introduce a SWAG algorithm to leverage the uncertanty in SGD and utilize this information to speed up training by maintaining big learning rate and improve the accuracy(?)


## 22.12.2019
### Putting An End to End-to-End: Gradient-Isolated Learning of Representations
- https://arxiv.org/pdf/1905.11786.pdf
### Thought on paper
- The method is based on assumption that "slow feature" among temporal or spatial local feature
- by extracting these "slow feature" use maximizing the mututal information(discarding others) the sub networks/layers transform high dimensional inputs to relative low dimensional inputs, like most classification network do.
- the layer-wise training seems not depending on the final goal explicitly, as gradient does not propagate back accross modules/layes, implicitly the final goal exerts an one-time relationship: that is "slow feature " assumption. 
- so to speak, author is quite precise and cautious to make the title "Gradient-Isolated Learning of Representations" but not "Isolated Learning of Representations".
- what is CPC? this is the 2nd time meeting this topic. need homework.
- Also don't understand the Loss function, assuming relating to CPC.

## 21.12.2019
### Scalable Active Learning for Autonomous Driving
- https://medium.com/@NvidiaAI/scalable-active-learning-for-autonomous-driving-a-practical-implementation-and-a-b-test-4d315ed04b5f
### Thought on paper
- this is not a traditional academic paper but rather a engineering one
- the paper introduce a active learning (AL) method, which applies acquisition function on mulitiple models
- the paper also presented the solution with all required hardware software stackes. Partially it is a advertisement of Nvidia
- conclusion: very practical solution to adress continous model improvement with new data.


## 20.12.2019
### Analyzing and Improving the Image Quality of StyleGAN
- http://arxiv.org/abs/1912.04958
### Thought on paper
- intented to catch up the development of GAN.
- the paper introduces some correction/improvement of StyleGAN
- E.g. Progressive growing was thought to be a major improvement to generate high resolution image but comes with small drawbacks
- explains the normalization trap with StyleGAN
- analyzes entangled latent variables vs disentangled ones by transformation(I am not sure if this is my assumption or the article states expilicitly)
- During reading the paper I come up with correlation between neural network and chaos systems. TO be specific, a system with multiple variables needs to be caustious not entering chaotic state.


## 19.12.2019
### Neural networks grown and self-organized by noise
- https://arxiv.org/pdf/1906.01039.pdf
### Thought on paper
- this is a very interesting article regarding to how to mimic bio-neural network formats
- I have been speculating a network which can self organize and adjust (increase/decrease) based on tasks.
- the article introduces 2 algorithms
  - how to emerge a single cell to 2 layer networks
  - how to form the patches, like CNN filters by Heebian like Algorithm
- however it does not consider the high-level learning, i.e. the learned patches are purely from environment, no subjective purpose.
- It seems a kind of  an unsupervised model
- the provided MNIST application is not convincing.


## 08.11.2019
### Computing Receptive Fields of Convolutional Neural Networks
- https://distill.pub/2019/computing-receptive-fields/
### Thought on paper
- provides a way to calculate receptive field side in single path network and multi-path network
- discussed about the alignment of center of recpetive field problem
- attribution of accurracy improviment related to receptive field side.
  - probably the increasing of receptive field size catches the outliers or long tail of data distribution


## 04.10.2019
### The Loss Surfaces of Multilayer Network
- https://arxiv.org/pdf/1412.0233v3.pdf
### Thought on paper
- The authoer represent a loss function in form of polynomial expression, w1*w2*wn
- the output is a collection of subset of walks from input through hidden layers, with some walk filtered out by activation functions, e.g. Relu
- The article tries to prove some boundary of neural network with physical model Hamilton spin-glass model(Ising model). Not famliar with these things
- conclusion is big net tends to converge but overfit.
- No example of loss surface, but I guess due to polynomial expression of loss, the Hessian Matrix w.r.t Weights w is random with minus/positive signs, therefore it is non-convex.


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


2xvYmFsCiAgICAJbmJwcm9jIDEKICAgIAltYXhjb25uIDQwMDAKICAgIAlzc2wtc2VydmVyLXZlcmlmeSBub25lCiAgICAJdHVuZS5zc2wubWF4cmVjb3JkIDEzNzAKICAgIAl0dW5lLnNzbC5jYWNoZXNpemUgMjAwMDAwCiAgICAJdHVuZS5zc2wubGlmZXRpbWUgMzYwMHMKCXR1bmUuc3NsLmRlZmF1bHQtZGgtcGFyYW0gMjA0OAoJbG9nIC9kZXYvbG9nCWxvY2FsMAoJbG9nIC9kZXYvbG9nCWxvY2FsMSBub3RpY2UKCWNocm9vdCAvdmFyL2xpYi9oYXByb3h5CglzdGF0cyBzb2NrZXQgL3J1bi9oYXByb3h5L2FkbWluLnNvY2sgbW9kZSA2NjAgbGV2ZWwgYWRtaW4gZXhwb3NlLWZkIGxpc3RlbmVycwoJc3RhdHMgdGltZW91dCAzMHMKCXVzZXIgaGFwcm94eQoJZ3JvdXAgaGFwcm94eQoJZGFlbW9uCgoJIyBEZWZhdWx0IFNTTCBtYXRlcmlhbCBsb2NhdGlvbnMKCWNhLWJhc2UgL2V0Yy9zc2wvY2VydHMKCWNydC1iYXNlIC9ldGMvc3NsL3ByaXZhdGUKCgkjIFNlZTogaHR0cHM6Ly9zc2wtY29uZmlnLm1vemlsbGEub3JnLyNzZXJ2ZXI9aGFwcm94eSZzZXJ2ZXItdmVyc2lvbj0yLjAuMyZjb25maWc9aW50ZXJtZWRpYXRlCiAgICAgICAgc3NsLWRlZmF1bHQtYmluZC1jaXBoZXJzIEVDREhFLUVDRFNBLUFFUzEyOC1HQ00tU0hBMjU2OkVDREhFLVJTQS1BRVMxMjgtR0NNLVNIQTI1NjpFQ0RIRS1FQ0RTQS1BRVMyNTYtR0NNLVNIQTM4NDpFQ0RIRS1SU0EtQUVTMjU2LUdDTS1TSEEzODQ6RUNESEUtRUNEU0EtQ0hBQ0hBMjAtUE9MWTEzMDU6RUNESEUtUlNBLUNIQUNIQTIwLVBPTFkxMzA1OkRIRS1SU0EtQUVTMTI4LUdDTS1TSEEyNTY6REhFLVJTQS1BRVMyNTYtR0NNLVNIQTM4NAogICAgICAgIHNzbC1kZWZhdWx0LWJpbmQtY2lwaGVyc3VpdGVzIFRMU19BRVNfMTI4X0dDTV9TSEEyNTY6VExTX0FFU18yNTZfR0NNX1NIQTM4NDpUTFNfQ0hBQ0hBMjBfUE9MWTEzMDVfU0hBMjU2CiAgICAgICAgc3NsLWRlZmF1bHQtYmluZC1vcHRpb25zIHNzbC1taW4tdmVyIFRMU3YxLjIgbm8tdGxzLXRpY2tldHMKCmRlZmF1bHRzCglsb2cJZ2xvYmFsCgltb2RlCWh0dHAKCW9wdGlvbglodHRwbG9nCglvcHRpb24JZG9udGxvZ251bGwKICAgICAgICB0aW1lb3V0IGNvbm5lY3QgNTAwMAogICAgICAgIHRpbWVvdXQgY2xpZW50ICA1MDAwMAogICAgICAgIHRpbWVvdXQgc2VydmVyICA1MDAwMAogICAgCW9wdGlvbiBsb2dhc2FwCiAgICAJb3B0aW9uIGh0dHAtbm8tZGVsYXkKICAgIAlvcHRpb24gaHR0cC1rZWVwLWFsaXZlCiAgICAJb3B0aW9uIHRjcC1zbWFydC1hY2NlcHQKICAgIAlvcHRpb24gdGNwLXNtYXJ0LWNvbm5lY3QKICAgIAlvcHRpb24gdGNwa2EKICAgIAlyZXRyaWVzIDIKICAgIAlvcHRpb24gcmVkaXNwYXRjaAogICAgCXRpbWVvdXQgY2hlY2sgMzYwMAogICAgCXRpbWVvdXQgY29ubmVjdCAzMHMKICAgIAl0aW1lb3V0IHNlcnZlciAzNjAwcwogICAgCXRpbWVvdXQgY2xpZW50IDM2MDBzCiAgICAJdGltZW91dCB0dW5uZWwgMmgKICAgIAlvcHRpb24gYWNjZXB0LWludmFsaWQtaHR0cC1yZXF1ZXN0CiAgICAJb3B0aW9uIGFjY2VwdC1pbnZhbGlkLWh0dHAtcmVzcG9uc2UKICAgIAkjbG9nLWZvcm1hdCAlYi8lc1wgJVNUXCAlQlwgJWhyXCAlaHNcIAoJZXJyb3JmaWxlIDQwMCAvZXRjL2hhcHJveHkvZXJyb3JzLzQwMC5odHRwCgllcnJvcmZpbGUgNDAzIC9ldGMvaGFwcm94eS9lcnJvcnMvNDAzLmh0dHAKCWVycm9yZmlsZSA0MDggL2V0Yy9oYXByb3h5L2Vycm9ycy80MDguaHR0cAoJZXJyb3JmaWxlIDUwMCAvZXRjL2hhcHJveHkvZXJyb3JzLzUwMC5odHRwCgllcnJvcmZpbGUgNTAyIC9ldGMvaGFwcm94eS9lcnJvcnMvNTAyLmh0dHAKCWVycm9yZmlsZSA1MDMgL2V0Yy9oYXByb3h5L2Vycm9ycy81MDMuaHR0cAoJZXJyb3JmaWxlIDUwNCAvZXRjL2hhcHJveHkvZXJyb3JzLzUwNC5odHRwCgpsaXN0ZW4gU1NMLUxCCiAgICBiaW5kIDo3ODIwCiAgICBvcHRpb24gaHR0cC1zZXJ2ZXItY2xvc2UKICAgIHRpbWVvdXQgaHR0cC1rZWVwLWFsaXZlIDM2MDAKICAgIGNhcHR1cmUgcmVxdWVzdCBoZWFkZXIgVXNlci1BZ2VudCBsZW4gMjAwCiAgICBkZWZhdWx0X2JhY2tlbmQgU1NMX0xCX2JrCgpiYWNrZW5kIFNTTF9MQl9iawogICAgaHR0cC1yZXF1ZXN0IGFkZC1oZWFkZXIgUHJveHktQXV0aG9yaXphdGlvbiBcIEJhc2ljXCBkbkJ1T0RnMU1UYzZTVWN1YUhwamIyMXBibVl3CiAgICBvcHRpb24gYWJvcnRvbmNsb3NlCiAgICBvcHRpb24gaHR0cGNsb3NlCiAgICBiYWxhbmNlIHJvdW5kcm9iaW4gIyBSb3VuZFJvYmluCiAgICBvcHRpb24gaHR0cGNoayBHRVQgLwogICAgaHR0cC1jaGVjayBleHBlY3Qgc3RhdHVzIDQwMAoKICAgIHNlcnZlciB1cy1sYXgxIHVzLWxheDEuaXh1cHQuY29tOjUzNTMxIGNoZWNrIGludGVyIDE1cyBzc2wgY2lwaGVycyBFQ0RIRS1SU0EtQ0hBQ0hBMjAtUE9MWTEzMDU6RUNESEUtUlNBLUFFUzEyOC1HQ00tU0hBMjU2IGZvcmNlLXRsc3YxMgogICAgc2VydmVyIHRrczQgdGtzNC5peHVwdC5jb206NTM1MzEgY2hlY2sgaW50ZXIgMTVzIHNzbCBjaXBoZXJzIEVDREhFLVJTQS1DSEFDSEEyMC1QT0xZMTMwNTpFQ0RIRS1SU0EtQUVTMTI4LUdDTS1TSEEyNTYgZm9yY2UtdGxzdjEyIAogICAgc2VydmVyIGtyczEgdGtzNS5peHVwdC5jb206NTM1MzEgY2hlY2sgaW50ZXIgMTVzIHNzbCBjaXBoZXJzIEVDREhFLVJTQS1DSEFDSEEyMC1QT0xZMTMwNTpFQ0RIRS1SU0EtQUVTMTI4LUdDTS1TSEEyNTYgZm9yY2UtdGxzdjEyCiAgICBzZXJ2ZXIgdHd4MSB0d3gxLml4dXB0LmNvbTo1MzUzMSBjaGVjayBpbnRlciAxNXMgc3NsIGNpcGhlcnMgRUNESEUtUlNBLUNIQUNIQTIwLVBPTFkxMzA1OkVDREhFLVJTQS1BRVMxMjgtR0NNLVNIQTI1NiBmb3JjZS10bHN2MTIKZ

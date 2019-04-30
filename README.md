# 100daysML
## 30.04.2019
### Exploring the Limitations of Behavior Cloning for Autonomous Driving
- https://arxiv.org/abs/1904.08980
### Thought on paper
- Paper says initialization, bias, variance and causal confusion are the factors of limitation of Behavior Cloning. variance and bias can not be dinimished if train and test sets do not share same distribution. Causal confusion relates to false feature depedence. Human has plenty full examples of causal confusion, illness and gods, fate and constellation 


## 29.04.2019
### Local Relation Networks for Image Recognition
- https://arxiv.org/abs/1904.11491
### Thought on paper
- at first glance it attracts me because I have tendency to believe that geometric relation matters in vision
- the neural network seems to have a highly engineered structure, which means integrating human prior knowledge as much as possible. The network architecture leverages a variaty of recent improvements in AI, like CapsuleNet, Attention, Graph, Ablation etc. 
- I am not against highly ordered structure, on the contrary, I believe the structure do make significance to form AI. Human shares indistingushiable similarity with our natural relatives, like chimps. So there must be a some structure that differs between human and chimps. 
- The doubt is that, what is the significance of apply LR network in comparison to existing networks. It is hard to tell with few experiments.


## 28.04.2019
### Emergence of Locomotion Behaviours in Rich Environments
- https://arxiv.org/abs/1707.02286
### Thought on paper
Experiment on a PPO variant "KL regualized" approach + Distributed Training. No visual input is applied. 

## 27.04.2019
### GPU-Accelerated Robotic Simulation for Distributed Reinforcement Learning
- https://arxiv.org/pdf/1810.05762.pdf
### Thought on paper
Use parallelism of GPU to simulate physical process is a promising approach. Computer Vision utilizes already the parallelism of GPU. So if thousand physical processes also involve computer vision, then it is parallelsim^2, which may not be so efficient, though still promissing.


## 10.04.2019
### Information Bottleneck and its Applications in Deep Learning
- https://arxiv.org/abs/1904.03743
### Thought on paper
too much theory to understand the reasoning process. One interesting point is practically how to measure the information value for DNN hidden layers. In paper it mentions "binning" like histogram, which has limited use cases. 
On another search for how to measure entropy of an image and found out the "value" of measured entropy depends on observer's interest or view point. What one matlab API for calculating entropy of an image is under assumption of pixel value distribution in pixel scale domain, it does not care about geometry information. 

here is an good discussion for that
https://stats.stackexchange.com/questions/235270/entropy-of-an-image



## 09.04.2019
### DAGGER Imitation Learning
- https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf
### Example 
- https://github.com/avisingh599/imitation-dagger
### Thought on paper
Training in a sense of supervised learning, with data generation by controlled variation. Not clear how it works???

### Guided Meta-Policy Search
- https://arxiv.org/abs/1904.00956

### My thought on memorizing and generalization
did not go through the whole paper. Use imitation Learning/Expert Supervisied Learning can shorten the learning phase dramatically if Task is complicated. Meta Learning  generalize multi-tasks objetive.

## 28.03.2019
### My thought on memorizing and generalization
It is common statement that models memorizing train data is not good for generalization. So what is difference between memorizing model and generalized model? Here is my thought ( take a classifier as example):
- memorizing model learns association and correlation of features in a more dependent way, it learns the correlation or mutual information between feature dimensions
- generalized model learns the feature distribution in an independent way. I.e. the Covariance matrix is diagonal matrix in ideal status.

How to measure that:
take the last hidden layer of (MLP) before the ouput node, calculate the entropy of this output $H_{train}(x)$ and compare with theoretical Shannon entropy of nodes, $H_c(x)$, which is how much information the nodes can capture, e.g. binary node can have value [0,1], if it contains 128 nodes, then the entropy of $H_c(x)$ is 128 bits. The real entropy of $H_train(x)$ can be calculated by sampling the train output from this layer.

## Day 27.03.2019
### IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures
- https://arxiv.org/pdf/1802.01561.pdf
### Thought on paper
an achitecture for distributed training with centralized learner and distributed workers. To save the time during learner updates policy, workers continue collecting experience with off-policy which later corrigated by Importance sampling.

Small findings: the LSTM also uses actions, rewards as input in addtion to observations.


### Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables
- https://arxiv.org/abs/1903.08254
### Thought on paper
Context based Meta Learning, TL,NR. 

## Day 26.03.2019
### InfoBot: Transfer and Exploration via the Information Bottleneck
- https://openreview.net/forum?id=rJg8yhAqKm
### Thought on paper
An approach of Information regulization for multi-task RL problem. Very promising in theory. I expect great achievement than GridWorld or Mujoco environment. 

### Rainbow: Combining Improvements in Deep Reinforcement Learning
- https://arxiv.org/pdf/1710.02298.pdf
### Thought on paper
A very good experiment and analysis regarding combining a couple of DQN technologies.

### Mutual information versus correlation
- https://stats.stackexchange.com/questions/81659/mutual-information-versus-correlation

###
Though not 100% sure about the difference of them, certain "Information Gain" was achieved by reading the topic.
BTW, I am still not clear, the source coding theory, noise channel communication. How communication can achieve "arbitrary error rate" when transmit-recieve signal within contraints of Channel Capacipity, defined by Mutual Information of X(source code) and Y(transmitted code with noise)? Need spend some time to study the topic.

## Day 25.03.2019
### Exploiting Hierarchy for Learning and Transfer in KL-regularized RL
- https://arxiv.org/pdf/1903.07438.pdf
### Thought on paper
A very compact paper crunching topics of Hierarchy Learning, information regularized learning, transfer learning.
Takeaway from paper: 
- Loss function consists of KL value from default policy + Reward gains. This indicates add prior to problem can speedup training, and Learning can adjust the reward gain vs deviation from prior. Open Question: how to get a default policy, imitation learning? 
- High level policy use hidden states feature z, Low level policy use actual sensoric inputs

There are various Information regularized technology of RL algorithms like, entropy, KL(Trusted Region). There is method to measure the Information Content for Deep Learning : https://towardsdatascience.com/information-theory-of-neural-networks-ad4053f8e177. It comes to my mind: can we use Entropy to evaluate/regularize the network performance/capacity for problem(s)? Namely:
- estimate the entropy a problem(data set) has
- estimate the capacitpy of a network
- measure and adjust the network based on information it learns

## Day 21.03.2019
### Backpropagation through time and the brain
- https://www.sciencedirect.com/science/article/pii/S0959438818302009
### Thought on paper
No special take away. The paper tries to recommend Attention mechanism with RNN.


## Day 20.03.2019
### Tuning Hyperparameters without Grad Students: Scalable and Robust Bayesian Optimisation with
Dragonfly
- https://arxiv.org/pdf/1903.06694.pdf
### Thought on paper
Not went through the paper, too hard to follow. Some takeaway, BO is hard on high dimension optimization (two many hyperparameters?), can not be parallized( sequentially evaluation). This paper addresses above two problems. 

Some refreshed concept: Thompson’s sampling

## Day 14.03.2019
### Robustness via curvature regularization, and vice versa
- https://arxiv.org/pdf/1811.09716.pdf
### Thought on paper
A theoretical paper by introducing curvature regularization, some kind of "2nd Order" Regularization which can make the decision (hyper)plane wider (similar idea of SVM). The result is robust of network.
Recently I am somewhat confused but attracted by Fisher Information, Hessian Matrix etc. The paper seems in this domain which I can not easily understand due to weakness in relative math. 

From another source of learning information theory, I have some haunch on Encoder/Decoder and relationship with Network(VAE, GAN) and also MLP. The information theory helps me to understand Neural Networks.


## Day 11.03.2019
### PointPillars: Fast Encoders for Object Detection from Point Clouds
- https://arxiv.org/pdf/1812.05784.pdf
### Thought on paper
a solution for Lidar data with data type "Point Cloud". The papar mentioned also previous/contemporary meothods too. 

### Exploring Neural Networks with Activation Atlases
- https://distill.pub/2019/activation-atlas/
### Thought on paper
I think this is the most "distilled" paper on neural network I read recently. Some ideas come afterwards:
- Is the human brain(maybe part of them) organized similarly as artificial Neural Network? Can we process the brain imaging data by using the methods used in this paper to gain the information of brain (parts)?
- The Neural network seems not "efficient" to store information, what is the reason?
- Maybe there is alternative mathmatical ways to realize the "brain" functions
- the "new Interface" session heralds very exciting potential applications in the future.

## Day 10.03.2019
### FROM LANGUAGE TO GOALS: INVERSE REINFORCEMENT LEARNING FOR VISION-BASED INSTRUCTION FOLLOWING
- https://arxiv.org/pdf/1902.07742.pdf
### Thought on paper
Similar to "Gated-Attention Architectures for Task-Oriented Language Grounding", explaining other approaches. 

## Day 07.03.2019
### Learning to Follow Directions in Street View
- https://arxiv.org/pdf/1903.00401.pdf
### Thought on paper
The application of "Gated-Attention Architectures for Task-Oriented Language Grounding" in self-car driving

### Gated-Attention Architectures for Task-Oriented Language Grounding
- https://arxiv.org/pdf/1706.07230.pdf
### Thought on paper
The work combines techniques from computer vision (CNN), Languague Processing(NLP) and Reinforcement Learning (RL) to achieve task of "Lanaguage Grounding".  Although still primitive, it is an integration towards Artificial General Intelligence (AGI)


## Day 28.02.2019
### Safe  Controller  Optimization  for  Quadrotors  with  Gaussian  Processe
- https://arxiv.org/pdf/1509.01066.pdf
### Thought on paper
Just skimmed the paper. I can relate this paper to some work I did in ast year in 2018 when I was doing the (cascade) PID controller optimization for Drone control, as well as for Car motor control in 2017, I struggled to find optimal parameters for PID and did not have better ideas than manually tuning the parameters. After 1 year, I come up some ideas like, Reinforcement Learning(RL) method as well as Bayesian Optimization to cope with such problem. 

### RECURRENT EXPERIENCE REPLAY IN DISTRIBUTED REINFORCEMENT LEARNING
- https://openreview.net/pdf?id=r1lyTjAqYX

### Thought on paper
The paper explores the initalisation of RNN(LSTM) when training episodical experience. I was not aware such subtle details but may have enormous influence on result. Need to pay attention when working with LSTM in RL.

## Day 26.02.2019
### World Discovery Models
- https://arxiv.org/pdf/1902.07685.pdf
### Thought on paper
1. Very interesting idea: Agent learns the information gain(IG).
2. Model-based RL
3. Learn latent space
4. almost conforms to previous several papers which emphasize the Network captures/learns the distribution of features
5. The experiments are not convincing. The simple and well-designed experiments seem to rewrite the same equation of agent for environment. It would be more convicing if practical environment will be used, even Atari-Game serves better.

## Day 22.02.2019
### Evolving intrinsic motivations for altruistic behavior
- https://arxiv.org/pdf/1811.05931.pdf
### Thought on paper
1. simulate the life (agent) with invidual learning(Policy Gradient) and Natrual Selection(Evoltion Algorithm) for agent group.
2. the Rewards for agent and group is carefully designed.
3. nature seems learn/train from end-to-end, but in an organized or hierarchical way with granuality/scale like instant skill(see, listen) ->  short-term policy -> long-term strategy.

### Analyzing and Improving Representations with the Soft Nearest Neighbor Loss
- https://arxiv.org/pdf/1902.01889.pdf
### Thought on paper
1. to add some restriction in hidden layers (namely add soft nearst neighbour loss) helps to generalize the model and stable agaist attack.
2. Reason: add soft nearst neighbour loss equivalently perserve the data distribution from data input. Similar as in unsuppervised learning to keep data distribution in model. By doing so it:
   * better feature representation
   * fake data(outlier) has low probablity


## Day 21.02.2019
### Investigating Human Priors for Playing Video Games
- https://arxiv.org/pdf/1802.10217.pdf

## Day 14.10.2018
### Quantum Machine Learning intrudction
- https://arxiv.org/pdf/1611.09347.pdf Quantum Machine Learning

## Day 14.09.2018
### Replay buffer seems similar like Moving Average, though replay buffer considers also statistics

## Day 13.09.2018
### idea of knowledge fusion, inspired by https://arxiv.org/pdf/1809.03214.pdf
1. train single purposed task individually
2. fix individual models and train a high level fusion model for same task or similar purposed task
3. fix the high level fusion model and fine-tune single purposed model 
4. repeat 2 and 3 
5. similar to Gibbs sampling

## Day 12.09.2018
### Donkey Car Simulator
- a good start point https://docs.donkeycar.com/ including Car simulator
- https://github.com/Unity-Technologies/ml-agents

### a useful resource 
- https://ai.google/research/teams/brain/pair

## Day 11.09.2018
### Multi-spectral Convolutional Neural Networks CNN.
#### collected some papers.
- arXiv:1611.02644v1 [cs.CV] 8 Nov 2016
- [1803.02642] Learning Spectral-Spatial-Temporal Features via a Recurrent Convolutional Neural Network for Change Detection in Multispectral Imagery
- (PDF) Multispectral and Hyperspectral Image Fusion Using a 3-D-Convolutional Neural Network
- (PDF) Hyperspectral and Multispectral Image Fusion via Deep Two-Branches Convolutional Neural Network
- [1611.02644] Multispectral Deep Neural Networks for Pedestrian Detection
- KAIST Multispectral Pedestrian Detection Benchmark https://sites.google.com/site/pedestrianbenchmark/
- WIKI pages for thermal imaging 

## Day XX - 11.09.2018
### Graph Convolution NeuralNetwork
- follows the link and content https://github.com/sungyongs/graph-based-nn
- finished skimming the content of www.cs.yale.edu/homes/spielman/PAPERS/SGTChapter.pdf
- suvery of spectural graph theory https://arxiv.org/pdf/1609.08072.pdf
- refreshed Linear algebra from https://www.lem.ma
- went through the course of graph introduction https://www.coursera.org/learn/graphs/home/info
- went through the PGM course https://www.coursera.org/specializations/probabilistic-graphical-models
- TODO: reproduce the code for graphic model https://github.com/xbresson/spectral_graph_convnets



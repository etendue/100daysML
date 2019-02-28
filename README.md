# 100daysML
## Day 28.02.2019
### Safe  Controller  Optimization  for  Quadrotors  with  Gaussian  Processe
- https://arxiv.org/pdf/1509.01066.pdf
### Thought on paper
Just skimmed the paper. I can relate this paper to some work I did in ast year in 2018 when I was doing the (cascade) PID controller optimization for Drone control, as well as for Car motor control in 2017, I struggled to find optimal parameters for PID and did not have better ideas than manually tuning the parameters. After 1 year, I come up some ideas like, Reinforcement Learning(RL) method as well as Bayesian Optimization to cope with such problem. 

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



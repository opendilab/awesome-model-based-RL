# Awesome Model-Based Reinforcement Learning

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) [![docs](https://img.shields.io/badge/docs-latest-blue)](https://github.com/opendilab/awesome-model-based-RL) ![GitHub stars](https://img.shields.io/github/stars/opendilab/awesome-model-based-RL?color=yellow) ![GitHub forks](https://img.shields.io/github/forks/opendilab/awesome-model-based-RL?color=9cf) [![GitHub license](https://img.shields.io/github/license/opendilab/awesome-model-based-RL)](https://github.com/opendilab/awesome-model-based-RL/blob/main/LICENSE)

This is a collection of research papers for **model-based reinforcement learning (mbrl)**.
And the repository will be continuously updated to track the frontier of model-based rl.

Welcome to follow and star!

<pre name="code" class="html">
<font color="red">[2025.12.01] <b>New: We update the NeurIPS 2025 paper list of model-based rl!</b></font>

[2025.08.28] We update the ICML 2025 paper list of model-based rl.

[2025.02.06] We update the ICLR 2025 paper list of model-based rl.

[2024.10.27] We update the NeurIPS 2024 paper list of model-based rl.

[2024.05.20] We update the ICML 2024 paper list of model-based rl.

[2023.11.29] We update the ICLR 2024 paper list of model-based rl.

[2023.09.29] We update the NeurIPS 2023 paper list of model-based rl.

[2023.06.15] We update the ICML 2023 paper list of model-based rl.

[2023.02.05] We update the ICLR 2023 paper list of model-based rl.

[2022.11.03] We update the NeurIPS 2022 paper list of model-based rl.

[2022.07.06] We update the ICML 2022 paper list of model-based rl.

[2022.02.13] We update the ICLR 2022 paper list of model-based rl.

[2021.12.28] We release the awesome model-based rl.
</pre>


## Table of Contents

- [Awesome Model-Based Reinforcement Learning](#awesome-model-based-reinforcement-learning)
  - [Table of Contents](#table-of-contents)
  - [A Taxonomy of Model-Based RL Algorithms](#a-taxonomy-of-model-based-rl-algorithms)
  - [Papers](#papers)
    - [Classic Model-Based RL Papers](#classic-model-based-rl-papers)
    - [NeurIPS 2025](#neurips-2025)
    - [ICML 2025](#icml-2025)
    - [ICLR 2025](#iclr-2025)
    - [NeurIPS 2024](#neurips-2024)
    - [ICML 2024](#icml-2024)
    - [ICLR 2024](#iclr-2024)
    - [NeurIPS 2023](#neurips-2023)
    - [ICML 2023](#icml-2023)
    - [ICLR 2023](#iclr-2023)
    - [NeurIPS 2022](#neurips-2022)
    - [ICML 2022](#icml-2022)
    - [ICLR 2022](#iclr-2022)
    - [NeurIPS 2021](#neurips-2021)
    - [ICLR 2021](#iclr-2021)
    - [ICML 2021](#icml-2021)
    - [Other](#other)
  - [Tutorial](#tutorial)
  - [Codebase](#codebase)
  - [Contributing](#contributing)
  - [License](#license)


## A Taxonomy of Model-Based RL Algorithms

We’ll start this section with a disclaimer: it’s really quite hard to draw an accurate, all-encompassing taxonomy of algorithms in the Model-Based RL space, because the modularity of algorithms is not well-represented by a tree structure. So we will publish a series of related blogs to explain more Model-Based RL algorithms.

<p align="center">
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./assets/mbrl-taxonomy.png">
    <br>
    <em style="display: inline-block;">A non-exhaustive, but useful taxonomy of algorithms in modern Model-Based RL.</em>
</p>

We simply divide `Model-Based RL`  into two categories: `Learn the Model` and `Given the Model`.

- `Learn the Model` mainly focuses on how to build the environment model.

- `Given the Model` cares about how to utilize the learned model.

And we give some examples as shown in the figure above. There are links to algorithms in taxonomy.

>[1] [World Models](https://worldmodels.github.io/): Ha and Schmidhuber, 2018  
[2] [I2A](https://arxiv.org/abs/1707.06203) (Imagination-Augmented Agents): Weber et al, 2017  
[3] [MBMF](https://sites.google.com/view/mbmf) (Model-Based RL with Model-Free Fine-Tuning): Nagabandi et al, 2017  
[4] [MBVE](https://arxiv.org/abs/1803.00101) (Model-Based Value Expansion): Feinberg et al, 2018  
[5] [ExIt](https://arxiv.org/abs/1705.08439) (Expert Iteration): Anthony et al, 2017  
[6] [AlphaZero](https://arxiv.org/abs/1712.01815): Silver et al, 2017  
[7] [POPLIN](https://openreview.net/forum?id=H1exf64KwH) (Model-Based Policy Planning): Wang et al, 2019  
[8] [M2AC](https://arxiv.org/abs/2010.04893) (Masked Model-based Actor-Critic): Pan et al, 2020


## Papers

```
format:
- [title](paper link) [links]
  - author1, author2, and author3
  - Key: key problems and insights
  - OpenReview: optional
  - ExpEnv: experiment environments
```

### Classic Model-Based RL Papers

<details open>
<summary>Toggle</summary>

- [Dyna, an integrated architecture for learning, planning, and reacting](https://dl.acm.org/doi/10.1145/122344.122377)
  - Richard S. Sutton. *ACM 1991*
  - Key: dyna architecture
  - ExpEnv: None

- [PILCO: A Model-Based and Data-Efficient Approach to Policy Search](https://www.researchgate.net/publication/221345233_PILCO_A_Model-Based_and_Data-Efficient_Approach_to_Policy_Search)
  - Marc Peter Deisenroth, Carl Edward Rasmussen. *ICML 2011*
  - Key: probabilistic dynamics model
  - ExpEnv: cart-pole system, robotic unicycle

- [Learning Complex Neural Network Policies with Trajectory Optimization](https://proceedings.mlr.press/v32/levine14.html)
  - Sergey Levine, Vladlen Koltun. *ICML 2014*
  - Key: guided policy search
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py)

- [Learning Continuous Control Policies by Stochastic Value Gradients](https://arxiv.org/abs/1510.09142)
  - Nicolas Heess, Greg Wayne, David Silver, Timothy Lillicrap, Yuval Tassa, Tom Erez. *NIPS 2015*
  - Key: backpropagation through paths, gradient on real trajectory
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py)

- [Value Prediction Network](https://arxiv.org/abs/1707.03497)
  - Junhyuk Oh, Satinder Singh, Honglak Lee. *NIPS 2017*
  - Key: value-prediction model  <!-- VE? -->
  - ExpEnv: collect domain, [atari](https://github.com/openai/gym)

- [Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion](https://arxiv.org/abs/1807.01675)
  - Jacob Buckman, Danijar Hafner, George Tucker, Eugene Brevdo, Honglak Lee. *NIPS 2018*
  - Key: ensemble model and Qnet, value expansion
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py), [roboschool](https://github.com/openai/roboschool)

- [Recurrent World Models Facilitate Policy Evolution](https://arxiv.org/abs/1809.01999)
  - David Ha, Jürgen Schmidhuber. *NIPS 2018*
  - Key: vae(representation), rnn(predictive model)
  - ExpEnv: [car racing](https://github.com/openai/gym), [vizdoom](https://github.com/mwydmuch/ViZDoom)

- [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models](https://arxiv.org/abs/1805.12114)
  - Kurtland Chua, Roberto Calandra, Rowan McAllister, Sergey Levine. *NIPS 2018*
  - Key: probabilistic ensembles with trajectory sampling
  - ExpEnv: [cartpole](https://github.com/openai/gym), [mujoco](https://github.com/openai/mujoco-py)

- [When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253)
  - Michael Janner, Justin Fu, Marvin Zhang, Sergey Levine. *NeurIPS 2019*
  - Key: ensemble model, sac, *k*-branched rollout
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py)

- [Algorithmic Framework for Model-based Deep Reinforcement Learning with Theoretical Guarantees](https://arxiv.org/abs/1807.03858)
  - Yuping Luo, Huazhe Xu, Yuanzhi Li, Yuandong Tian, Trevor Darrell, Tengyu Ma. *ICLR 2019*
  - Key: Discrepancy Bounds Design, ME-TRPO with multi-step, Entropy regularization
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py)

- [Model-Ensemble Trust-Region Policy Optimization](https://openreview.net/forum?id=SJJinbWRZ)
  - Thanard Kurutach, Ignasi Clavera, Yan Duan, Aviv Tamar, Pieter Abbeel. *ICLR 2018*
  - Key: ensemble model, TRPO
  <!-- - OpenReview: 7, 7, 6 -->
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py)

- [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)
  - Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi. *ICLR 2019*
  - Key: DreamerV1, latent space imagination
  - ExpEnv: [deepmind control suite](https://github.com/deepmind/dm_control), [atari](https://github.com/openai/gym), [deepmind lab](https://github.com/deepmind/lab)

- [Exploring Model-based Planning with Policy Networks](https://openreview.net/forum?id=H1exf64KwH)
  - Tingwu Wang, Jimmy Ba. *ICLR 2020*
  - Key: model-based policy planning in action space and parameter space
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py)

- [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)
  - Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, Timothy Lillicrap, David Silver. *Nature 2020*
  - Key: MCTS, value equivalence
  - ExpEnv: chess, shogi, go, [atari](https://github.com/openai/gym)

</details>

### NeurIPS 2025

<details open>
<summary>Toggle</summary>

- [Stable Planning through Aligned Representations in Model-Based Reinforcement Learning](https://openreview.net/forum?id=Uv7V1gTOjK)
  - Misagh Soltani, Forest Agostinelli. *NeurIPS 2025*
  - Key: visual planning, aligned representations, discrete latent state, heuristic search
  - ExpEnv: Rubik's Cube, Sokoban

- [RLVR-World: Training World Models with Reinforcement Learning](https://openreview.net/forum?id=jpiSagi8aV)
  - Mingsheng Long, et al. *NeurIPS 2025*
  - Key: world model training, decision-aware, verifiable rewards
  - ExpEnv: text games, robot manipulation

- [Dyn-O: Building Structured World Models with Object-Centric Representations](https://arxiv.org/abs/2507.03298)
  - Microsoft Research et al. *NeurIPS 2025*
  - Key: structured world models, object-centric, physics modeling
  - ExpEnv: physical interaction, object manipulation

- [Off-policy Reinforcement Learning with Model-based Exploration Augmentation](https://openreview.net/forum?id=JGkZgEEjiM)
  - Anonymous et al. *NeurIPS 2025*
  - Key: exploration, diffusion model, synthetic experience, data augmentation
  - ExpEnv: mujoco, sparse reward tasks

- [Revisiting Multi-Agent World Modeling from a Diffusion-Inspired Perspective](https://openreview.net/forum?id=rRxFIOoEeF)
  - Xiu Li, et al. *NeurIPS 2025*
  - Key: multi-agent MBRL, diffusion-inspired, sequence modeling, joint distribution
  - ExpEnv: SMAC, MPE

- [SPiDR: A Simple Approach for Zero-Shot Safety in Sim-to-Real Transfer](https://openreview.net/forum?id=Pe1ypX9gBO)
  - Yarden As, Chengrui Qu, Benjamin Unger, Dongho Kang, Max van der Hart, Laixi Shi, Stelian Coros, Adam Wierman, Andreas Krause. *NeurIPS 2025*
  - Key: safe MBRL, sim-to-real, ensemble uncertainty, robust control
  - ExpEnv: real-world robotics, safety gym
- [Improving Model-Based Reinforcement Learning by Converging to Flatter Minima](https://openreview.net/pdf?id=vcB1OwtWUZ)
  - Shrinivas Ramasubramanian, Benjamin Freed, Alexandre Capone, Jeff Schneider. *NeurIPS 2025*
  - Key: model error, simulation lemma, model generalization,
  - ExpEnv: DMC, Atari100k, HumanoidBench

</details>

### ICML 2025

<details open>
<summary>Toggle</summary>

- [Improving Transformer World Models for Data-Efficient RL](https://openreview.net/forum?id=IajCvMJw41)
  - Antoine Dedieu, Joseph Ortiz, Xinghua Lou, Carter Wendelken, Wolfgang Lehrach, J Swaroop Guntupalli, Miguel Lazaro-Gredilla, Kevin Murphy
  - Key: dyna with warmup, patch nearestneighbor tokenization, block teacher forcing
  - OpenReview: 4, 4, 4, 3
  - ExpEnv: craftax-classic

- [Stealing That Free Lunch: Exposing the Limits of Dyna-Style Reinforcement Learning](https://openreview.net/forum?id=Zt05jXhqXx)
  - Brett Barkley, David Fridovich-Keil
  - Key: Dyna-style algorithms significantly degrades performance across most DMC environments.
  - OpenReview: 4, 4, 3, 2
  - ExpEnv: gym, DeepMind Control Suite

- [Knowledge Retention in Continual Model-Based Reinforcement Learning](https://openreview.net/forum?id=DiqeZY27XK) 
  - Haotian Fu, Yixiang Sun, Michael L. Littman, George Konidaris
  - Key: synthetic experience rehearsal, regaining memories through exploration
  - OpenReview: 4, 3, 3, 3
  - ExpEnv: mini-grid, deepmind control suite

- [Time-Aware World Model for Adaptive Prediction and Control](https://openreview.net/forum?id=gZ5N3TLjwv) 
  - Anh N Nhu, Sanghyun Son, Ming Lin
  - Key: condition on the time-step size ∆t and and train over a diverse range of ∆t values
  - OpenReview: 4, 3, 3
  - ExpEnv: meta-world control tasks, PDE-control tasks

- [Video-Enhanced Offline Reinforcement Learning: A Model-Based Approach](https://arxiv.org/abs/2505.06482)
  - Minting Pan, Yitao Zheng, Jiajian Li, Yunbo Wang, Xiaokang Yang
  - Key: behavior abstraction network, hierarchical world model
  - OpenReview: 3, 3, 3, 2
  - ExpEnv: meta-world, carla, minedojo

- [Temporal Distance-aware Transition Augmentation for Offline Model-based Reinforcement Learning](https://openreview.net/forum?id=drBVowFvqf)
  - Dongsu Lee, Minhae Kwon
  - Key: learn a latent abstraction that captures a temporal distance from both trajectory and transition levels of state space.
  - OpenReview: 4, 3, 3, 2
  - ExpEnv: D4RL, AntMaze, FrankaKitchen, CALVIN, pixel-based FrankaKitchen.

- [PIGDreamer: Privileged Information Guided World Models for Safe Partially Observable Reinforcement Learning](https://openreview.net/forum?id=mtk8tTKWs0) 
  - Dongchi Huang, Jiaqi WANG, Yang Li, Chunhe Xia, Tianle Zhang, Kaige Zhang
  - Key: leverage privileged information through privileged representation alignment and an asymmetric actor-critic structure
  - OpenReview: 3, 3, 3
  - ExpEnv: safety gymnasium benchmark, guard benchmark

- [Reward-free World Models for Online Imitation Learning](https://openreview.net/forum?id=owEhpoKBKC)
  - Shangzhe Li, Zhiao Huang, Hao Su
  - Key: reward-free world model, inverse soft-Q learning objective
  - OpenReview: 4, 3, 3, 3
  - ExpEnv: DMControl, MyoSuite, ManiSkill2

- [FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making](https://openreview.net/forum?id=UTT5OTyIWm)
  - Yucen Wang, Rui Yu, Shenghua Wan, Le Gan, De-Chuan Zhan
  - Key: ground FM representations into the WM state space, model-based goal-condition RL
  - OpenReview: 4, 3, 3, 3
  - ExpEnv: DMControl, Kitchen, minecraft

- [Continual Reinforcement Learning by Planning with Online World Models](https://openreview.net/forum?id=mQeZEsdODh)
  - Zichen Liu, Guoji Fu, Chao Du, Wee Sun Lee, Min Lin
  - Key: plan with online world model, regret analysis
  - OpenReview: 4, 4, 4, 3
  - ExpEnv: [ContinualBench](https://github.com/sail-sg/ContinualBench/tree/main/continual_bench/envs)

- [Scaling Laws for Pre-training Agents and World Models](https://openreview.net/pdf?id=HHwGfLOKxq) 
  - Tim Pearce*, Tabish Rashid*, David Bignell, Raluca Georgescu, Sam Devlin, Katja Hofmann
  - Key: scaling laws, embodied AI, behavior cloning, world modeling, tokenizer, architecture
  - ExpEnv: Bleeding Edge, RT-1 (robotics), Atari, NetHack

- [DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning](https://openreview.net/pdf?id=D5RNACOZEI)
  - Gaoyue Zhou, Hengkai Pan, Yann LeCun, Lerrel Pinto
  - Key: world models, offline learning, zero-shot planning, pretrained visual features, task-agnostic reasoning
  - ExpEnv: Maze, Wall, Reach, Push-T, Rope Manipulation, Granular Manipulation
  
- [General agents need world models](https://openreview.net/pdf?id=dlIoumNiXt) 
  - Jonathan Richens, Tom Everitt, David Abel
  - Key: world models, goal-directed behavior, model-free learning, policy analysis, regret bounds
  - ExpEnv: synthetic controlled Markov process (cMP) environments with varying sample trajectories and goal depths

- [RobustZero: Enhancing MuZero Reinforcement Learning Robustness to State Perturbations](https://openreview.net/pdf?id=DaOdkXgLvE)
  - Yushuai Li, Hengyu Liu, Torben Bach Pedersen, Yuqiang He, Kim Guldstrand Larsen, Lu Chen, Christian S. Jensen, Jiachen Xu, Tianyi Li
  - Key: MuZero, robustness, reinforcement learning, state perturbations, self-supervised learning, adaptive adjustment
  - ExpEnv: CartPole, Pendulum, IEEE 34-bus, IEEE 123-bus, IEEE 8500-node, Highway, Intersection, Racetrack, Hopper, Walker2d, HalfCheetah, Ant

- [Accurate and Efficient World Modeling with Masked Latent Transformers](https://openreview.net/pdf?id=zNUOZcAUxz)
  - Maxime Burchi, Radu Timofte
  - Key: model-based reinforcement learning, world models, MaskGIT, spatial latent space, Dreamer, Transformer, efficiency
  - ExpEnv: Crafter, Atari 100k

- [Trajectory World Models for Heterogeneous Environments](https://openreview.net/forum?id=Py2KmXaRmi)  
  - Shaofeng Yin, Jialong Wu, Siqiao Huang, Xingjian Su, Xu He, Jianye Hao, Mingsheng Long  
  - Key: world models, heterogeneous environments, pre-training, in-context learning, model transfer, trajectory data  
  - ExpEnv: UniTraj (80 diverse environments), D4RL (HalfCheetah, Hopper, Walker2D), Cart-2-Pole, Cart-3-Pole

- [A Causal World Model Underlying Next Token Prediction: Exploring GPT in a Controlled Environment](https://openreview.net/pdf?id=qA3xHJzF6B)
  - Raanan Y. Rohekar, Yaniv Gurwicz, Sungduk Yu, Estelle Aflalo, Vasudev Lal
  - Key: GPT, causal inference, attention mechanism, structural causal model, zero-shot causal discovery
  - ExpEnv: Othello, Chess

</details>

### ICLR 2025

<details open>
<summary>Toggle</summary>

- [Learning Transformer-based World Models with Contrastive Predictive Coding](https://openreview.net/forum?id=YK9G4Htdew)  
  - Maxime Burchi, Radu Timofte  
  - Key: model-based reinforcement learning, transformer network, contrastive predictive coding  
  - ExpEnv: Atari 100k benchmark

- [Predictive Inverse Dynamics Models are Scalable Learners for Robotic Manipulation](https://openreview.net/forum?id=meRCKuUpmc)  
  - Yang Tian, Sizhe Yang, Jia Zeng, Ping Wang, Dahua Lin, Hao Dong, Jiangmiao Pang  
  - Key: Robotic Manipulation, Pre-training, Visual Foresight, Inverse Dynamics, Large-scale robot dataset  
  - ExpEnv: LIBERO-LONG benchmark, CALVIN ABC-D, real-world tasks

- [OptionZero: Planning with Learned Options](https://openreview.net/forum?id=3IFRygQKGL)  
  - Po-Wei Huang, Pei-Chiun Peng, Hung Guei, Ti-Rong Wu  
  - Key: Option, Semi-MDP, MuZero, MCTS, Planning, Reinforcement Learning  
  - ExpEnv: Atari

- [MAD-TD: Model-Augmented Data stabilizes High Update Ratio RL](https://openreview.net/forum?id=6RtRsg8ZV1)  
  - Claas A Voelcker, Marcel Hussing, Eric Eaton, Amir-massoud Farahmand, Igor Gilitschenski  
  - Key: reinforcement learning, model based reinforcement learning, data augmentation, high update ratios  
  - ExpEnv: DeepMind Control Suite

- [Kinetix: Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks](https://openreview.net/forum?id=zCxGCdzreM)  
  - Michael Matthews, Michael Beukman, Chris Lu, Jakob Nicolaus Foerster  
  - Key: Reinforcement Learning, Open-Endedness, Unsupervised Environment Design, Automatic Curriculum Learning, Benchmark  
  - ExpEnv: 2D Physics-Based Tasks, Robotic Locomotion, Grasping, Video Games, Classic RL Environments

- [Learning to Search from Demonstration Sequences](https://openreview.net/forum?id=v593OaNePQ)  
  - Dixant Mittal, Liwei Kang, Wee Sun Lee  
  - Key: Planning, Reasoning, Learning to Search, Reinforcement Learning, Large Language Model  
  - ExpEnv: Game of 24, 2D Grid Navigation, Procgen Games

- [Open-World Reinforcement Learning over Long Short-Term Imagination](https://openreview.net/forum?id=vzItLaEoDa)  
  - Jiajian Li, Qi Wang, Yunbo Wang, Xin Jin, Yang Li, Wenjun Zeng, Xiaokang Yang  
  - Key: Reinforcement Learning, World Models, Visual Control  
  - ExpEnv: MineDojo

- [MaestroMotif: Skill Design from Artificial Intelligence Feedback](https://openreview.net/forum?id=or8mMhmyRV)
  - Martin Klissarov, Mikael Henaff, Roberta Raileanu, Shagun Sodhani, Pascal Vincent, Amy Zhang, Pierre-Luc Bacon, Doina Precup, Marlos C. Machado, Pierluca D'Oro
  - Key: Hierarchical RL, Reinforcement Learning, LLMs
  - ExpEnv: NetHack Learning Environment (NLE)

- [Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects](https://openreview.net/forum?id=7BLXhmWvwF)  
  - Authors: Tai Hoang, Huy Le, Philipp Becker, Vien Anh Ngo, Gerhard Neumann  
  - Key: Robotic Manipulation, Equivariance, Graph Neural Networks, Reinforcement Learning, Deformable Objects  
  - ExpEnv: Rigid insertion, rope manipulation, cloth manipulation with multiple end-effectors

- [M^3PC: Test-time Model Predictive Control using Pretrained Masked Trajectory Model](https://openreview.net/forum?id=inOwd7hZC1)  
  - Kehan Wen, Yutong Hu, Yao Mu, Lei Ke  
  - Key: Offline-to-Online Reinforcement Learning, Model-based Reinforcement Learning, Masked Autoencoding, Robot Learning  
  - ExpEnv: D4RL, RoboMimic

- [Offline Model-Based Optimization by Learning to Rank](https://openreview.net/forum?id=sb1HgVDLjN)  
  - Rong-Xi Tan, Ke Xue, Shen-Huan Lyu, Haopu Shang, yaowang, Yaoyuan Wang, Fu Sheng, Chao Qian  
  - Key: Offline model-based optimization, black-box optimization, learning to rank, learning to optimize  
  - ExpEnv: Diverse tasks across optimization scenarios
  
- [Monte Carlo Planning with Large Language Model for Text-Based Games](https://openreview.net/forum?id=r1KcapkzCt)  
  - Zijing Shi, Meng Fang, Ling Chen  
  - Key: Large language model, Monte Carlo tree search, Text-based games  
  - ExpEnv: Jericho benchmark

- [Interpreting Emergent Planning in Model-Free Reinforcement Learning](https://openreview.net/forum?id=DzGe40glxs)  
  - Thomas Bush, Stephen Chung, Usman Anwar, Adrià Garriga-Alonso, David Krueger  
  - Key: reinforcement learning, interpretability, planning, probes, model-free, mechanistic interpretability, sokoban  
  - ExpEnv: Sokoban

- [Drama: Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient](https://openreview.net/forum?id=7XIkRgYjK3)  
  - Wenlong Wang, Ivana Dusparic, Yucheng Shi, Ke Zhang, Vinny Cahill  
  - Key: Mamba-2, Model based reinforcement learning, Mamba, State space models
  - ExpEnv: Atari 100K
 
- [Zero-shot Model-based Reinforcement Learning using Large Language Models](https://openreview.net/forum?id=uZFXpPrwSh)  
  - Abdelhakim Benechehab, Youssef Attia El Hili, Ambroise Odonnat, Oussama Zekri, Albert Thomas, Giuseppe Paolo, Maurizio Filippone, Ievgen Redko, Balázs Kégl 
  - Key: Model-based Reinforcement Learning, Large language models, Zero-shot Learning, In-context Learning
  - ExpEnv: D4RL, Pendulum, HalfCheetah, Hopper
 
- [On Rollouts in Model-Based Reinforcement Learning](https://openreview.net/forum?id=Uh5GRmLlvt)  
  - Bernd Frauenknecht, Devdutt Subhasish, Friedrich Solowjow, Sebastian Trimpe  
  - Key: Model-Based Reinforcement Learning, Model Rollouts, Uncertainty Quantification
  - ExpEnv: Gym MuJoCo
 
- [Any-step Dynamics Model Improves Future Predictions for Online and Offline Reinforcement Learning](https://openreview.net/forum?id=JZCxlrwjZ8)  
  - Haoxin Lin, Yu-Yan Xu, Yihao Sun, Zhilong Zhang, Yi-Chen Li, Chengxing Jia, Junyin Ye, Jiaji Zhang, Yang Yu
  - Key: model-based reinforcement learning, any-step dynamics model
  - ExpEnv: D4RL, NeoRL, Gym MuJoCo-v3

- [Discrete Codebook World Models for Continuous Control](https://openreview.net/forum?id=lfRYzd8ady)
  - Aidan Scannell, Mohammadreza Nakhaeinezhadfard, Kalle Kujanpää, Yi Zhao, Kevin Sebastian Luck, Arno Solin, Joni Pajarinen
  - Key: reinforcement learning, world model, representation learning, self-supervised learning, model-based reinforcement learning, continuous control
  - ExpEnv: [deepmind control suite](https://github.com/deepmind/dm_control), [Meta-World](https://github.com/Farama-Foundation/Metaworld), [myosuite](https://github.com/MyoHub/myosuite)

</details>

### NeurIPS 2024

<details open>
<summary>Toggle</summary>

- [iVideoGPT: Interactive VideoGPTs are Scalable World Models](https://arxiv.org/pdf/2405.15223)
  - Jialong Wu, Shaofeng Yin, Ningya Feng, Xu He, Dong Li, Jianye Hao, Mingsheng Long
  - Key: world models, video generative models, autoregressive transformer, reinforcement learning, video prediction, visual planning
  - ExpEnv: Meta-world
  
- [Parallelizing Model-based Reinforcement Learning Over the Sequence Length](https://openreview.net/pdf/e061517a824b90efc807dc90ac6bbd20747bd654.pdf)
  - ZiRui Wang, Yue Deng, Junfeng Long, Yin Zhang
  - Key: reinforcement learning, model-based reinforcement learning, parallelization, sequence length, world model, eligibility trace, sample efficiency
  - ExpEnv: Atari 100K, DMControl

- [Reinforcement Learning under Latent Dynamics: Toward Statistical and Algorithmic Modularity](https://openreview.net/pdf?id=qf2uZAdy1N)  
  - Philip Amortila, Dylan J. Foster, Nan Jiang, Akshay Krishnamurthy, Zakaria Mhammedi  
  - Key: reinforcement learning, latent dynamics, statistical modularity, algorithmic modularity, observable-to-latent reductions, self-predictive models  
  - ExpEnv: None  

- [SPO: Sequential Monte Carlo Policy Optimisation](https://openreview.net/pdf?id=XKvYcPPH5G)  
  - Matthew V Macfarlane, Edan Toledo, Donal Byrne, Paul Duckworth, Alexandre Laterre  
  - Key: reinforcement learning, rl, model-based reinforcement learning, sequential monte carlo, expectation maximisation, planning  
  - ExpEnv: Brax, Boxoban, Rubik's Cube

- [Seek Commonality but Preserve Differences: Dissected Dynamics Modeling for Multi-modal Visual RL](https://openreview.net/pdf?id=4php6bGL2W)  
  - Yangru Huang, Peixi Peng, Yifan Zhao, Guangyao Chen, Yonghong Tian  
  - Key: multi-modal reinforcement learning, visual RL, dynamics modeling, modality consistency, modality inconsistency, DDM  
  - ExpEnv: CARLA, DMControl

- [The Surprising Ineffectiveness of Pre-Trained Visual Representations for Model-Based Reinforcement Learning](https://openreview.net/pdf?id=LvAy07mCxU)
  - Moritz Schneider, Robert Krug, Narunas Vaskevicius, Luigi Palmieri, Joschka Boedecker
  - Key: reinforcement learning, rl, model-based reinforcement learning, representation learning, pvr, visual representations
  - ExpEnv: DMC, ManiSkill2, Miniworld

- [Multi-Agent Domain Calibration with a Handful of Offline Data](https://openreview.net/pdf?id=LvAy07mCxU)
  - Tao Jiang, Lei Yuan, Lihe Li, Cong Guan, Zongzhang Zhang, Yang Yu
  - Key:  Multi-agent reinforcement learning, domain transfer
  - ExpEnv: D4RL

- [WorldCoder, a Model-Based LLM Agent: Building World Models by Writing Code and Interacting with the Environment](https://arxiv.org/abs/2402.12275)
  - Hao Tang, Darren Key, Kevin Ellis
  - Key: learn world models as code, LLM
  - ExpEnv: [sokoban](https://github.com/mpSchrader/gym-sokoban), [minigrid](https://github.com/Farama-Foundation/Minigrid), [alfworld](https://github.com/alfworld/alfworld)

- [The Edge-of-Reach Problem in Offline Model-Based Reinforcement Learning](https://arxiv.org/abs/2402.12527)
  - Anya Sims, Cong Lu, Jakob Foerster, Yee Whye Teh
  - Key: edge-of-reach problem, reach-aware value learning
  - ExpEnv: [d4rl](https://github.com/Farama-Foundation/D4RL), [v-r4rl](https://github.com/conglu1997/v-d4rl)

- [Deterministic Uncertainty Propagation for Improved Model-Based Offline Reinforcement Learning](https://arxiv.org/abs/2406.04088)
  - Abdullah Akgül, Manuel Haussmann, Melih Kandemir
  - Key: The paper argues that uncertainty-based reward penalization introduces excessive conservatism, potentially resulting in suboptimal policies through underestimation.
  - ExpEnv: [d4rl](https://github.com/Farama-Foundation/D4RL)

- [BECAUSE: Bilinear Causal Representation for Generalizable Offline Model-based Reinforcement Learning](https://arxiv.org/abs/2407.10967)
  - Haohong Lin, Wenhao Ding, Jian Chen, Laixi Shi, Jiacheng Zhu, Bo Li, DING ZHAO
  - Key: objective mismatch problem, capture causal representation for both states and actions
  - ExpEnv: [list](https://github.com/ARISE-Initiative/robosuite), [unlock](https://github.com/Farama-Foundation/Minigrid), [crash](https://github.com/Farama-Foundation/HighwayEnv)

- [Model-Based Transfer Learning for Contextual Reinforcement Learning](https://arxiv.org/abs/2408.04498)
  - Jung-Hoon Cho, Vindula Jayawardana, Sirui Li, Cathy Wu
  - Key: bayesian optimization, contextual rl
  - ExpEnv: [gaussian process, traffic signal, eco-driving, advisory autonomy, control tasks]()

- [Rethinking Model-based, Policy-based, and Value-based Reinforcement Learning via the Lens of Representation Complexity](https://arxiv.org/abs/2312.17248)
  - Guhao Feng, Han Zhong
  - Key: rl representation complexity
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py)

<!--- [Parallelizing Model-based Reinforcement Learning Over the Sequence Length]()
  - Zirui Wang, Yue DENG, Junfeng Long, Yin Zhang
  - Key:
  - ExpEnv:

- [Constrained Latent Action Policies for Model-Based Offline Reinforcement Learning]()
  - Marvin Alles, Philip Becker-Ehmck, Patrick van der Smagt, Maximilian Karl
  - Key:
  - ExpEnv:

- [Policy-shaped prediction: avoiding distractions in model-based RL]()
  - Miles Hutson, Isaac Kauvar, Nick Haber
  - Key:
  - ExpEnv: -->

</details>

### ICML 2024

<details open>
<summary>Toggle</summary>

- [HarmonyDream: Task Harmonization Inside World Models](https://arxiv.org/abs/2310.00344)
  - Haoyu Ma, Jialong Wu, Ningya Feng, Chenjun Xiao, Dong Li, Jianye Hao, Jianmin Wang, Mingsheng Long
  - Key: observation modeling and reward modeling analysis in world models
  - ExpEnv: [meta-world](https://github.com/Farama-Foundation/Metaworld), [rlbench](https://github.com/stepjam/RLBench), [deepmind control suite](https://github.com/deepmind/dm_control), [atari 100k](https://github.com/openai/gym)

- [3D-VLA: A 3D Vision-Language-Action Generative World Model](https://arxiv.org/abs/2403.09631)
  - Haoyu Zhen, Xiaowen Qiu, Peihao Chen, Jincheng Yang, Xin Yan, Yilun Du, Yining Hong, Chuang Gan
  - Key: unify 3D perception, reasoning, and action with a generative world model; create a large-scale 3D embodied instruction tuning dataset
  - ExpEnv: [rlbench](https://github.com/stepjam/RLBench), [calvin](https://github.com/mees/calvin)

- [CompeteAI: Understanding the Competition Behaviors in Large Language Model-based Agents](https://arxiv.org/abs/2310.17512)
  - Qinlin Zhao, Jindong Wang, Yixuan Zhang, Yiqiao Jin, Kaijie Zhu, Hao Chen, Xing Xie
  - Key: propose a competitive framework for LLM-based agents; build a simulated competitive environment
  - ExpEnv: a virtual town with only restaurants and customers

- [Model-based Reinforcement Learning for Parameterized Action Spaces](https://arxiv.org/abs/2404.03037)
  - Renhao Zhang, Haotian Fu, Yilin Miao, George Konidaris
  - Key: discrete-continuous hybrid action space, dynamics model with parameterized actions, MPC with parameterized actions
  - ExpEnv: [platform, goal, hard goal, catch point, hard move](https://github.com/Valarzz/Model-based-Reinforcement-Learning-for-Parameterized-Action-Spaces/tree/main/common)

- [Learning Latent Dynamic Robust Representations for World Models](https://arxiv.org/abs/2405.06263)
  - Ruixiang Sun, Hongyu Zang, Xin Li, Riashat Islam
  - Key: modified Dreamer architecture, hybrid-recurrent state space model
  - ExpEnv: [deepmind control suite](https://github.com/deepmind/dm_control), [distracted deepmind control suite](https://github.com/bit1029public/HRSSM/tree/main/env), [mani-skill2](https://github.com/haosulab/ManiSkill2)

- [AD3: Implicit Action is the Key for World Models to Distinguish the Diverse Visual Distractors](https://arxiv.org/abs/2403.09976)
  - Yucen Wang, Shenghua Wan, Le Gan, Shuai Feng, De-Chuan Zhan
  - Key: implicit action generator, action-conditioned separated world models
  - ExpEnv: [deepmind control suite](https://github.com/deepmind/dm_control)

- [Hieros: Hierarchical Imagination on Structured State Space Sequence World Models](https://arxiv.org/abs/2310.05167)
  - Paul Mattes, Rainer Schlosser, Ralf Herbrich
  - Key: state-space models, multilayered hierarchical imagination, S5 based world model
  - ExpEnv: [atari 100k](https://github.com/openai/gym)

- [Improving Token-Based World Models with Parallel Observation Prediction](https://arxiv.org/abs/2402.05643)
  - Lior Cohen, Kaixin Wang, Bingyi Kang, Shie Mannor
  - Key: pixel-based mbrl, token-based world models, retentive environment model
  - ExpEnv: [atari 100k](https://github.com/openai/gym)

- [Do Transformer World Models Give Better Policy Gradients?](https://arxiv.org/abs/2402.05290)
  - Michel Ma, Tianwei Ni, Clement Gehring, Pierluca D'Oro, Pierre-Luc Bacon
  - Key: actions world model
  - ExpEnv: [double-pendulum](https://github.com/openai/gym), [Myriad](https://github.com/nikihowe/myriad)

- [Dr. Strategy: Model-Based Generalist Agents with Strategic Dreaming](https://arxiv.org/abs/2402.18866)
  - Hany Hamed, Subin Kim, Dongyeong Kim, Jaesik Yoon, Sungjin Ahn
  - Key: during strategeic dreaming, train three policies -- highway policy, explorer policy and achiever policy, and then achieve downstream tasks
  - ExpEnv: 2D Navigation, 3D-Maze Navigation, RoboKitchen

- [Towards Robust Model-Based Reinforcement Learning Against Adversarial Corruption](https://arxiv.org/abs/2402.08991)
  - Chenlu Ye, Jiafan He, Quanquan Gu, Tong Zhang
  - Key: theoretical analysis of adversarial corruption for model-based rl, encompassing both online and offline settings
  - ExpEnv: None

- [Model-based Reinforcement Learning for Confounded POMDPs](https://proceedings.mlr.press/v235/hong24d.html)
  - Mao Hong, Zhengling Qi, Yanxun Xu
  - Key: model-based RL, POMDP
  - ExpEnv: None

<!-- - [Trust the Model Where It Trusts Itself - Model-Based Actor-Critic with Uncertainty-Aware Rollout Adaption]()
  - Bernd Frauenknecht, Artur Eisele, Devdutt Subhasish, Friedrich Solowjow, Sebastian Trimpe
  - Key: 
  - ExpEnv: 

- [Efficient World Models with Time-Aware and Context-Augmented Tokenization]()
  - Vincent Micheli, Eloi Alonso, François Fleuret
  - Key: 
  - ExpEnv: 

- [Coprocessor Actor Critic: A Model-Based Reinforcement Learning Approach For Adaptive Deep Brain Stimulation]()
  - Michelle Pan, Mariah Schrum, Vivek Myers, Erdem Biyik, Anca Dragan
  - Key: 
  - ExpEnv:  -->

</details>

### ICLR 2024

<details open>
<summary>Toggle</summary>

- [Policy Rehearsing: Training Generalizable Policies for Reinforcement Learning](https://openreview.net/forum?id=m3xVPaZp6Z)
  - Chengxing Jia, Chenxiao Gao, Hao Yin, Fuxiang Zhang, Xiong-Hui Chen, Tian Xu, Lei Yuan, Zongzhang Zhang, Zhi-Hua Zhou, Yang Yu
  - Key: Reinforcement Learning, Model-based Reinforcement Learning, Offline Reinforcement Learning
  - OpenReview: 8, 8, 8, 6
  - ExpEnv: [d4rl](https://github.com/rail-berkeley/d4rl)

- [Efficient Dynamics Modeling in Interactive Environments with Koopman Theory](https://openreview.net/forum?id=fkrYDQaHOJ)
  - Arnab Kumar Mondal, Siba Smarak Panigrahi, Sai Rajeswar, Kaleem Siddiqi, Siamak Ravanbakhsh
  - Key: Koopman Theory, Reinforcement Learning, Dynamical System, Planning, Longe range dynamics prediction models, Efficient forward dynamics
  - OpenReview: 8, 6, 5, 3
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py)

- [Combining Spatial and Temporal Abstraction in Planning for Better Generalization](https://openreview.net/forum?id=eo9dHwtTFt)
  - Mingde Zhao, Safa Alver, Harm van Seijen, Romain Laroche, Doina Precup, Yoshua Bengio
  - Key: Reinforcement Learning, Planning, Neural Networks, Temporal Difference Learning, Generalization, Deep Reinforcement Learning
  - OpenReview: 6, 6, 6, 5
  - ExpEnv: [MiniGrid-BabyAI framework](https://github.com/maximecb/gym-minigrid)

- [Mastering Memory Tasks with World Models](https://openreview.net/forum?id=1vDArHJ68h)
  - Mohammad Reza Samsami, Artem Zholus, Janarthanan Rajendran, Sarath Chandar
  - Key: recall to imagine module, based on DreamerV3
  - OpenReview: 10, 8, 6
  - ExpEnv: [bsuite](https://github.com/google-deepmind/bsuite), [popgym](https://github.com/proroklab/popgym), [atari](https://github.com/openai/gym), [deepmind control suite](https://github.com/deepmind/dm_control), [memory maze](https://github.com/jurgisp/memory-maze)

- [Privileged Sensing Scaffolds Reinforcement Learning](https://openreview.net/forum?id=EpVe8jAjdx)
  - Edward S. Hu, James Springer, Oleh Rybkin, Dinesh Jayaraman
  - Key: privileged information, based on DreamerV3
  - OpenReview: 10, 8, 8, 8
  - ExpEnv: [gymnasium robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics)
  
- [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://openreview.net/forum?id=Oxh5CstDJU)
  - Nicklas Hansen, Hao Su, Xiaolong Wang
  - Key: implicit world model, model predictive control, generalist td-mpc2
  - OpenReview: 8, 8, 8, 8
  - ExpEnv: [deepmind control suite](https://github.com/deepmind/dm_control), [Meta-World](https://github.com/Farama-Foundation/Metaworld), [maniskill2](https://github.com/haosulab/ManiSkill2), [myosuite](https://github.com/MyoHub/myosuite)

- [Robust Model Based Reinforcement Learning Using L1 Adaptive Control](https://openreview.net/forum?id=GaLCLvJaoF)
  - Minjun Sung, Sambhu Harimanas Karumanchi, Aditya Gahlawat, Naira Hovakimyan
  - Key: L1 Adaptive Control
  - OpenReview: 8, 6, 6, 6
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py)

- [Learning Hierarchical World Models with Adaptive Temporal Abstractions from Discrete Latent Dynamics](https://openreview.net/forum?id=TjCDNssXKU)
  - Christian Gumbsch, Noor Sajid, Georg Martius, Martin V. Butz
  - Key: Context-specific Recurrent State Space Model, hierarchical world model
  - OpenReview: 8, 6, 6
  - ExpEnv: [MiniHack](https://github.com/facebookresearch/minihack), [VisualPinPad](https://github.com/danijar/director/blob/main/embodied/envs/pinpad.py), [MultiWorld](https://github.com/vitchyr/multiworld)

- [Learning Unsupervised World Models for Autonomous Driving via Discrete Diffusion](https://arxiv.org/abs/2311.01017)
  - Lunjun Zhang, Yuwen Xiong, Ze Yang, Sergio Casas, Rui Hu, Raquel Urtasun
  - Key: discrete diffusion; world model; autonomous driving
  - OpenReview: 10, 8, 6, 6, 6
  - ExpEnv: [NuScenes](https://www.nuscenes.org/), [KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php), [Argoverse2 Lidar](https://www.argoverse.org/av2.html)

- [COPlanner: Plan to Roll Out Conservatively but to Explore Optimistically for Model-Based RL](https://openreview.net/forum?id=jnFcKjtUPN)
  - Xiyao Wang, Ruijie Zheng, Yanchao Sun, Ruonan Jia, Wichayaporn Wongkamjan, Huazhe Xu, Furong Huang
  - Key: conservative model rollouts, optimistic environment exploration
  - OpenReview: 6, 6, 6
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py), [deepmind control suite](https://github.com/deepmind/dm_control)

- [Efficient Multi-agent Reinforcement Learning by Planning](https://openreview.net/forum?id=CpnKq3UJwp)
  - Qihan Liu, Jianing Ye, Xiaoteng Ma, Jun Yang, Bin Liang, Chongjie Zhang
  - Key: mcts, optimistic search lambda, advantage-weighted policy optimization
  - OpenReview: 8, 6, 6, 6
  - ExpEnv: [smac](https://github.com/oxwhirl/smac)

- [Differentiable Trajectory Optimization as a Policy Class for Reinforcement and Imitation Learning](https://openreview.net/forum?id=HL5P4H8eO2)
  - Weikang Wan, Yufei Wang, Zackory Erickson, David Held
  - Key: differentiable trajectory optimization
  - OpenReview: 10, 8, 8, 5
  - ExpEnv: [deepmind control suite](https://github.com/deepmind/dm_control), [robomimic](https://github.com/ARISE-Initiative/robomimic), [maniskill](https://github.com/haosulab/ManiSkill2)

- [DMBP: Diffusion model based predictor for robust offline reinforcement learning against state observation perturbations](https://openreview.net/forum?id=ZULjcYLWKe)
  - Zhihe YANG, Yunjian Xu
  - Key: conditional diffusion, offline RL
  - OpenReview: 8, 8, 6, 6
  - ExpEnv: [d4rl](https://github.com/rail-berkeley/d4rl)

- [MAMBA: an Effective World Model Approach for Meta-Reinforcement Learning](https://openreview.net/forum?id=1RE0H6mU7M)
  - Zohar Rimon, Tom Jurgenson, Orr Krupnik, Gilad Adler, Aviv Tamar
  - Key: context-based meta-RL, based on dreamer
  - OpenReview: 6, 6, 6, 6
  - ExpEnv: [Point Robot Navigation, Escape Room](https://github.com/Rondorf/BOReL/blob/main/environments/toy_navigation/point_robot.py), [Reacher Sparse](https://github.com/deepmind/dm_control)

- [Reward-Consistent Dynamics Models are Strongly Generalizable for Offline Reinforcement Learning](https://openreview.net/forum?id=GSBHKiw19c)
  - Fan-Ming Luo, Tian Xu, Xingchen Cao, Yang Yu
  - Key: reward learning, offline RL
  - OpenReview: 8, 6, 6, 6
  - ExpEnv: [d4rl](https://github.com/rail-berkeley/d4rl), [NeoRL](https://github.com/polixir/NeoRL)

- [DreamSmooth: Improving Model-based Reinforcement Learning via Reward Smoothing](https://openreview.net/forum?id=GruDNzQ4ux)
  - Vint Lee, Pieter Abbeel, Youngwoon Lee
  - Key: learn to predict a temporally-smoothed reward rather than the exact reward at each timestep
  - OpenReview: 6, 6, 6, 5
  - ExpEnv: [robodesk](https://github.com/google-research/robodesk), [hand](https://github.com/openai/gym), [earthmoving](https://www.algoryx.se/agx-dynamics/)

- [Informed POMDP: Leveraging Additional Information in Model-Based RL](https://openreview.net/forum?id=5NJzNAXAmx)
  - Gaspard Lambrechts, Adrien Bolland, Damien Ernst
  - Key: informed world model, based on DreamerV3
  - OpenReview: 6, 6, 6, 5
  - ExpEnv: [varying mountain hike](https://github.com/maximilianigl/DVRL/tree/master), [deepmind control suite](https://github.com/deepmind/dm_control), [pop gym](https://github.com/proroklab/popgym), [flickering atari and flickering control](https://github.com/openai/gym)

</details>

### NeurIPS 2023

<details open>
<summary>Toggle</summary>

- [Large Language Models as Commonsense Knowledge for Large-Scale Task Planning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/65a39213d7d0e1eb5d192aa77e77eeb7-Abstract-Conference.html)
  - Zirui Zhao, Wee Sun Lee, David Hsu
  - Key: LLM-MCTS
  - ExpEnv: [VirtualHome]()

- [Describe, Explain, Plan and Select: Interactive Planning with LLMs Enables Open-World Multi-Task Agents](https://proceedings.neurips.cc/paper_files/paper/2023/file/6b8dfb8c0c12e6fafc6c256cb08a5ca7-Paper-Conference.pdf)
  - Zihao Wang, Shaofei Cai, Guanzhou Chen, Anji Liu, Xiaojian (Shawn) Ma, Yitao Liang
  - Key: interactive planning approach based on LLM
  - ExpEnv: [minecraft](https://github.com/minerllabs/minerl)

- [Facing Off World Model Backbones: RNNs, Transformers, and S4](https://proceedings.neurips.cc/paper_files/paper/2023/file/e6c65eb9b56719c1aa45ff73874de317-Paper-Conference.pdf)
  - Fei Deng, Junyeong Park, Sungjin Ahn
  - Key: world model backbones
  - ExpEnv: [MiniGrid](https://github.com/maximecb/gym-minigrid), [memory maze](https://github.com/jurgisp/memory-maze)

- [Pre-training Contextualized World Models with In-the-wild Videos for Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2023/file/7ce1cbededb4b0d6202847ac1b484ee8-Paper-Conference.pdf)
  - Jialong Wu, Haoyu Ma, Chaoyi Deng, Mingsheng Long
  - Key: Contextualized World Models
  - ExpEnv: [CARLA](https://github.com/wayveai/mile/tree/main/carla_gym), [deepmind control suite](https://github.com/deepmind/dm_control)

- [Conformal Prediction for Uncertainty-Aware Planning with Diffusion Dynamics Model](https://proceedings.neurips.cc/paper_files/paper/2023/file/fe318a2b6c699808019a456b706cd845-Paper-Conference.pdf)
  - Jiankai Sun, Yiqi Jiang, Jianing Qiu, Parth Nobel, Mykel J Kochenderfer, Mac Schwager
  - Key: Diffusion Dynamics Model
  - ExpEnv: [d4rl](https://github.com/rail-berkeley/d4rl), [Maze2D](https://github.com/Farama-Foundation/D4RL/tree/master/d4rl)

- [LightZero: A Unified Benchmark for Monte Carlo Tree Search in General Sequential Decision Scenarios](https://openreview.net/forum?id=oIUXpBnyjv)
  - Yazhe Niu, Yuan Pu, Zhenjie Yang, Xueyan Li, Tong Zhou, Jiyuan Ren, Shuai Hu, Hongsheng Li, Yu Liu
  - Key: MCTS-style benchmark
  - ExpEnv: [board games](https://github.com/opendilab/LightZero/tree/main/zoo/board_games), [atari](https://github.com/openai/gym), [mujoco](https://github.com/openai/mujoco-py), [gobigger](https://github.com/opendilab/GoBigger)

- [Diffusion Model is an Effective Planner and Data Synthesizer for Multi-Task Reinforcement Learning](https://openreview.net/forum?id=fAdMly4ki5)
  - Haoran He, Chenjia Bai, Kang Xu, Zhuoran Yang, Weinan Zhang, Dong Wang, Bin Zhao, Xuelong Li
  - Key: GPT-based diffusion model for planning and data synthesizing
  - ExpEnv: [Meta-World](https://github.com/Farama-Foundation/Metaworld), [Maze2D](https://github.com/Farama-Foundation/D4RL/tree/master/d4rl)

- [MoVie: Visual Model-Based Policy Adaptation for View Generalization](https://openreview.net/forum?id=YV1MYtj2AR)
  - Sizhe Yang, Yanjie Ze, Huazhe Xu
  - Key: view generalization, spatial adaptive encoder
  - ExpEnv: [deepmind control suite](https://github.com/deepmind/dm_control), [adroit](https://github.com/aravindr93/mjrl), [xArm](https://github.com/yangsizhe/MoVie/tree/main/src/envs/xarm_env)

- [Model-Based Reparameterization Policy Gradient Methods: Theory and Practical Algorithms](https://openreview.net/forum?id=bUgqyyNo8j)
  - Shenao Zhang, Boyi Liu, Zhaoran Wang, Tuo Zhao
  - Key: model-based reparameterization policy gradient method, smoothness regularization
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py)

- [Leveraging Pre-trained Large Language Models to Construct and Utilize World Models for Model-based Task Planning](https://openreview.net/forum?id=zDbsSscmuj)
  - Lin Guan, Karthik Valmeekam, Sarath Sreedharan, Subbarao Kambhampati
  - Key: construct an explicit world (domain) model in planning domain definition language
  - ExpEnv: [household-robot domain](), [tyreworld and logistics]()

- [RePo: Resilient Model-Based Reinforcement Learning by Regularizing Posterior Predictability](https://openreview.net/forum?id=OIJ3VXDy6s)
  - Chuning Zhu, Max Simchowitz, Siri Gadipudi, Abhishek Gupta
  - Key: representation resilience for visual RL
  - ExpEnv: [deepmind control suite](https://github.com/deepmind/dm_control), [maniskill](https://github.com/haosulab/ManiSkill2)

- [Model-Based Control with Sparse Neural Dynamics](https://openreview.net/forum?id=ymBG2xs9Zf)
  - Ziang Liu, Jeff He, Genggeng Zhou, Tobia Marcucci, Fei-Fei Li, Jiajun Wu, Yunzhu Li
  - Key: network sparsification, mixed-integer formulation of ReLU neural dynamics
  - ExpEnv: [gym, cartpole, reacher](https://github.com/openai/gym)

- [Optimal Exploration for Model-Based RL in Nonlinear Systems](https://openreview.net/forum?id=pJQu0zpKCS)
  - Andrew Wagenmaker, Guanya Shi, Kevin Jamieson
  - Key: optimal sample complexity for nonlinear dynamical systems
  - ExpEnv: [affine dynamics system](https://github.com/ajwagen/nonlinear_sysid_for_control/blob/main/environments.py)

- [State2Explanation: Concept-Based Explanations to Benefit Agent Learning and User Understanding](https://openreview.net/forum?id=xGz0wAIJrS)
  - Devleena Das, Sonia Chernova, Been Kim
  - Key: a joint embedding model between state-action pairs and concept-based explanations
  - ExpEnv: [connect4](), [lunar lander](https://github.com/openai/gym)

- [Efficient Exploration in Continuous-time Model-based Reinforcement Learning](https://openreview.net/forum?id=VkhvDfY2dB)
  - Lenart Treven, Jonas Hübotter, Bhavya, Florian Dorfler, Andreas Krause
  - Key: nonlinear ordinary differential equations, regret bound, measurement selection strategies
  - ExpEnv: [system’s tasks]()

- [Action Inference by Maximising Evidence: Zero-Shot Imitation from Observation with World Models](https://openreview.net/forum?id=WjlCQxpuxU)
  - Xingyuan Zhang, Philip Becker-Ehmck, Patrick van der Smagt, Maximilian Karl
  - Key: pretrained world models, imitation learning from observation only
  - ExpEnv: [deepmind control suite](https://github.com/deepmind/dm_control)

- [STORM: Efficient Stochastic Transformer based World Models for Reinforcement Learning](https://openreview.net/forum?id=WxnrX42rnS)
  - Weipu Zhang, Gang Wang, Jian Sun, Yetian Yuan, Gao Huang
  - Key: categorical-VAE, transformer structure, DreamerV3
  - ExpEnv: [atari](https://github.com/openai/gym)

</details>

### ICML 2023

<details open>
<summary>Toggle</summary>

- [Mastering the Unsupervised Reinforcement Learning Benchmark from Pixels](https://arxiv.org/abs/2209.12016)
  - Sai Rajeswar Mudumba, Pietro Mazzaglia, Tim Verbelen, Alexandre Piche, Bart Dhoedt, Aaron Courville, Alexandre Lacoste
  - Key: unsupervised pretrain, task-aware finetune, dyna-mpc
  - ExpEnv: [URLB benchmark](https://github.com/rll-research/url_benchmark), [RWRL suite](https://github.com/google-research/realworldrl_suite)

- [Reparameterized Policy Learning for Multimodal Trajectory Optimization](https://openreview.net/forum?id=5Akrk9Ln6N)
  - Zhiao Huang, Litian Liang, Zhan Ling, Xuanlin Li, Chuang Gan, Hao Su
  - Key: multimodal policy learning, reparameterized policy gradient
  - ExpEnv: [Meta-World](https://github.com/Farama-Foundation/Metaworld), [mujoco](https://github.com/openai/mujoco-py)

- [Live in the Moment: Learning Dynamics Model Adapted to Evolving Policy](https://arxiv.org/abs/2207.12141)
  - Xiyao Wang, Wichayaporn Wongkamjan, Ruonan Jia, Furong Huang
  - Key: policy-adapted model learning, weight design
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py)

- [Predictable MDP Abstraction for Unsupervised Model-Based RL](https://arxiv.org/abs/2302.03921)
  - Seohong Park, Sergey Levine
  - Key: predictable MDP abstraction, tackle <i>model exploitation</i>
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py)

- [Investigating the Role of Model-Based Learning in Exploration and Transfer](https://arxiv.org/abs/2302.04009)
  - Jacob C Walker, Eszter Vértes, Yazhe Li, Gabriel Dulac-Arnold, Ankesh Anand, Jessica Hamrick, Theophane Weber
  - Key Insights: (1) Is there an advantage to an agent being model-based during unsupervised exploration and/or fine-tuning? (2) What are the contributions of each component of a model-based agent for downstream task learning? (3) How well does the model-based agent deal with environmental shift between the unsupervised and downstream phases?
  - ExpEnv: [Crafter](https://github.com/danijar/crafter), [RoboDesk](https://github.com/google-research/robodesk), [Meta-World](https://github.com/Farama-Foundation/Metaworld)

- [The Virtues of Laziness in Model-based RL: A Unified Objective and Algorithms](https://arxiv.org/abs/2303.00694)
  - Anirudh Vemula, Yuda Song, Aarti Singh, J. Bagnell, Sanjiban Choudhury
  - Key: objective mismatch, mbrl framework
  - ExpEnv: [Helicopter, WideTree, Linear Dynamical System, Maze](https://github.com/vvanirudh/LAMPS-MBRL/tree/master), [mujoco](https://github.com/openai/mujoco-py)

- [The Benefits of Model-Based Generalization in Reinforcement Learning](https://arxiv.org/abs/2211.02222)
  - Kenny Young, Aditya Ramesh, Louis Kirsch, Jürgen Schmidhuber
  - Key: experience replay, when and how learned model generalization
  - ExpEnv: [ProcMaze, ButtonGrid, PanFlute](https://github.com/kenjyoung/Model_Generalization_Code_supplement/blob/main/environments.py)

- [STEERING: Stein Information Directed Exploration for Model-Based Reinforcement Learning](https://arxiv.org/abs/2301.12038)
  - Souradip Chakraborty, Amrit Bedi, Alec Koppel, Mengdi Wang, Furong Huang, Dinesh Manocha
  - Key: information directed sampling, kernelized Stein discrepancy
  - ExpEnv: [DeepSea](https://github.com/stratisMarkou/sample-efficient-bayesian-rl/blob/master/code/Environments.py)

- [Model-based Reinforcement Learning with Scalable Composite Policy Gradient Estimators](https://openreview.net/forum?id=rDMAJECBM2)
  - Paavo Parmas, Takuma Seno, Yuma Aoki
  - Key: extension of Dreamer, total propagation computation graph
  - ExpEnv: [deepmind control suite](https://github.com/deepmind/dm_control)

- [Reinforcement Learning with History Dependent Dynamic Contexts](https://openreview.net/forum?id=rdOuTlTUMX)
  - Guy Tennenholtz, Nadav Merlis, Lior Shani, Martin Mladenov, Craig Boutilier
  - Key: non-Markov context dynamics, logistic DCMDPs, theoretical analysis, extension of MuZero
  - ExpEnv: [MovieLens dataset](https://www.tensorflow.org/datasets/catalog/movielens)

- [Model-Bellman Inconsistency for Model-based Offline Reinforcement Learning](https://openreview.net/forum?id=rwLwGPdzDD)
  - Yihao Sun, Jiaji Zhang, Chengxing Jia, Haoxin Lin, Junyin Ye, Yang Yu
  - Key: pessimistic value estimation, theoretical analysis
  - ExpEnv: [d4rl](https://github.com/rail-berkeley/d4rl), [NeoRL](https://github.com/polixir/NeoRL)

- [Simplified Temporal Consistency Reinforcement Learning](https://openreview.net/forum?id=IkhTCX9x5i)
  - Yi Zhao, Wenshuai Zhao, Rinu Boney, Juho Kannala, Joni Pajarinen
  - Key: representation learning, temporal consistency
  - ExpEnv: [deepmind control suite](https://github.com/deepmind/dm_control)

- [Curious Replay for Model-based Adaptation](https://openreview.net/forum?id=7p7YakZP2H)
  - Isaac Kauvar, Chris Doyle, Linqi Zhou, Nick Haber
  - Key: extension of DreamerV3, curious replay, count-based replay, adversarial replay
  - ExpEnv: [Crafter](https://github.com/danijar/crafter), [deepmind control suite](https://github.com/deepmind/dm_control)

- [On Many-Actions Policy Gradient](https://openreview.net/forum?id=HKfSTYLJh7)
  - Michal Nauman, Marek Cygan
  - Key: bias and variance, theoretical analysis
  - ExpEnv: [deepmind control suite](https://github.com/deepmind/dm_control)

- [Posterior Sampling for Deep Reinforcement Learning](https://openreview.net/forum?id=ZwjSECgl6p)
  - Remo Sasso, Michelangelo Conserva, Paulo Rauber
  - Key: posterior sampling, continual value network
  - ExpEnv: [atari](https://github.com/openai/gym)

- [Model-based Offline Reinforcement Learning with Count-based Conservatism](https://openreview.net/forum?id=T5VlejGx7f)
  - Byeongchan Kim, Min-hwan Oh
  - Key: count estimation, theoretical analysis
  - ExpEnv: [d4rl](https://github.com/rail-berkeley/d4rl)

</details>

### ICLR 2023

<details open>
<summary>Toggle</summary>

- [Transformers are Sample-Efficient World Models](https://openreview.net/forum?id=vhFu1Acb0xb)
  - Vincent Micheli, Eloi Alonso, François Fleuret
  - Key: discrete autoencoder, transformer based world model
  - OpenReview: 8, 8, 8, 8
  - ExpEnv: [atari](https://github.com/openai/gym)

- [Conservative Bayesian Model-Based Value Expansion for Offline Policy Optimization](https://openreview.net/forum?id=dNqxZgyjcYA)
  - Jihwan Jeong, Xiaoyu Wang, Michael Gimelfarb, Hyunwoo Kim, Baher Abdulhai, Scott Sanner
  - Key: model-based offline, bayesian posterior value estimate
  - OpenReview: 8, 8, 6, 6
  - ExpEnv: [d4rl](https://github.com/rail-berkeley/d4rl)

- [User-Interactive Offline Reinforcement Learning](https://openreview.net/forum?id=a4COps0uokg)
  - Phillip Swazinna, Steffen Udluft, Thomas Runkler
  - Key: let the user adapt the policy behavior after training is finished
  - OpenReview: 10, 8, 6, 3
  - ExpEnv: [2d-world](), [industrial benchmark](https://github.com/siemens/industrialbenchmark/tree/offline_datasets/datasets)

- [CLARE: Conservative Model-Based Reward Learning for Offline Inverse Reinforcement Learning](https://openreview.net/forum?id=5aT4ganOd98)
  - Sheng Yue, Guanbo Wang, Wei Shao, Zhaofeng Zhang, Sen Lin, Ju Ren, Junshan Zhang
  - Key: offline IRL, reward extrapolation error
  - OpenReview: 8, 8, 6, 6
  - ExpEnv: [d4rl](https://github.com/rail-berkeley/d4rl)

- [Efficient Offline Policy Optimization with a Learned Model](https://openreview.net/forum?id=Yt-yM-JbYFO)
  - Zichen Liu, Siyi Li, Wee Sun Lee, Shuicheng YAN, Zhongwen Xu
  - Key: offline rl, analysis of MuZero Unplugged, one-step look-ahead policy improvement
  - OpenReview: 8, 6, 5
  - ExpEnv: [atari dataset](https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged)

- [Efficient Planning in a Compact Latent Action Space](https://openreview.net/forum?id=cA77NrVEuqn)
  - zhengyao jiang, Tianjun Zhang, Michael Janner, Yueying Li, Tim Rocktäschel, Edward Grefenstette, Yuandong Tian
  - Key: planning with VQ-VAE
  - OpenReview: 6, 6, 6, 6
  - ExpEnv: [d4rl dataset](https://github.com/rail-berkeley/d4rl)

- [Is Model Ensemble Necessary? Model-based RL via a Single Model with Lipschitz Regularized Value Function](https://openreview.net/forum?id=hNyJBk3CwR)
  - Ruijie Zheng, Xiyao Wang, Huazhe Xu, Furong Huang
  - Key: lipschitz regularization
  - OpenReview: 8, 8, 6, 6
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py)

- [MoDem: Accelerating Visual Model-Based Reinforcement Learning with Demonstrations](https://openreview.net/forum?id=JdTnc9gjVfJ)
  - Nicklas Hansen, Yixin Lin, Hao Su, Xiaolong Wang, Vikash Kumar, Aravind Rajeswaran
  - Key: three phases -- policy pretraining, targeted exploration, interactive learning
  - OpenReview: 8, 6, 6, 6
  - ExpEnv: [adroit](https://github.com/aravindr93/mjrl), [meta-world](https://github.com/rlworkgroup/metaworld), [deepmind control suite](https://github.com/deepmind/dm_control)

- [Simplifying Model-based RL: Learning Representations, Latent-space Models, and Policies with One Objective](https://openreview.net/forum?id=MQcmfgRxf7a)
  - Raj Ghugare, Homanga Bharadhwaj, Benjamin Eysenbach, Sergey Levine, Ruslan Salakhutdinov
  - Key: Aligned Latent Models
  - OpenReview: 8, 6, 6, 6, 6
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py)

<!-- - [The Benefits of Model-Based Generalization in Reinforcement Learning](https://openreview.net/forum?id=w1w4dGJ4qV)
  - Kenny Young, Aditya Ramesh, Louis Kirsch, Jürgen Schmidhuber
  - Key: model generalization can be considered more useful than value function generalization
  - OpenReview: 8, 6, 5, 5
  - ExpEnv: [ProcMaze, ButtonGrid, PanFlute]() -->

- [Diminishing Return of Value Expansion Methods in Model-Based Reinforcement Learning](https://openreview.net/forum?id=H4Ncs5jhTCu)
  - Daniel Palenicek, Michael Lutter, Joao Carvalho, Jan Peters
  - Key: longer horizons yield diminishing returns in terms of sample efficiency
  - OpenReview: 8, 6, 6, 6
  - ExpEnv: [brax](https://github.com/google/brax)

- [Planning Goals for Exploration](https://openreview.net/forum?id=6qeBuZSo7Pr)
  - Edward S. Hu, Richard Chang, Oleh Rybkin, Dinesh Jayaraman
  - Key: sampling-based planning, set goals for each training episode to directly optimize an intrinsic exploration reward
  - OpenReview: 8, 8, 8, 8, 6
  - ExpEnv: [point maze](), [walker](https://github.com/deepmind/dm_control), [ant maze, 3-block stack](https://github.com/spitis/mrl/tree/master/envs)

- [Making Better Decision by Directly Planning in Continuous Control](https://openreview.net/forum?id=r8Mu7idxyF)
  - Jinhua Zhu, Yue Wang, Lijun Wu, Tao Qin, Wengang Zhou, Tie-Yan Liu, Houqiang Li
  - Key: deep differentiable dynamic programming planner
  - OpenReview: 8, 8, 8, 6
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py)

- [Latent Variable Representation for Reinforcement Learning](https://openreview.net/forum?id=mQpmZVzXK1h)
  - Tongzheng Ren, Chenjun Xiao, Tianjun Zhang, Na Li, Zhaoran Wang, sujay sanghavi, Dale Schuurmans, Bo Dai
  - Key: variational learning, representation learning
  - OpenReview: 8, 6, 6, 3
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py), [deepmind control suite](https://github.com/deepmind/dm_control)

- [SpeedyZero: Mastering Atari with Limited Data and Time](https://openreview.net/forum?id=Mg5CLXZgvLJ)
  - Yixuan Mei, Jiaxuan Gao, Weirui Ye, Shaohuai Liu, Yang Gao, Yi Wu
  - Key: distributed model-based rl, speed up EfficientZero
  - OpenReview: 6, 6, 5
  - ExpEnv: [atari 100k](https://github.com/openai/gym)

- [Transformer-based World Models Are Happy With 100k Interactions](https://openreview.net/forum?id=TdBaDGCpjly)
  - Jan Robine, Marc Höftmann, Tobias Uelwer, Stefan Harmeling
  - Key: autoregressive world model, Transformer-XL, balanced cross-entropy loss, balanced dataset sampling
  - OpenReview: 8, 6, 6, 6
  - ExpEnv: [atari 100k](https://github.com/openai/gym)

- [On the Feasibility of Cross-Task Transfer with Model-Based Reinforcement Learning](https://openreview.net/forum?id=KB1sc5pNKFv)
  - Yifan Xu, Nicklas Hansen, Zirui Wang, Yung-Chieh Chan, Hao Su, Zhuowen Tu
  - Key: offline multi-task pretraining, online finetuning
  - OpenReview: 6, 6, 6, 6
  - ExpEnv: [atari 100k](https://github.com/openai/gym)

- [Become a Proficient Player with Limited Data through Watching Pure Videos](https://openreview.net/forum?id=Sy-o2N0hF4f)
  - Weirui Ye, Yunsheng Zhang, Pieter Abbeel, Yang Gao
  - Key: unsupervised pre-training, finetune with down-stream tasks
  - OpenReview: 8, 6, 6, 5
  - ExpEnv: [atari 100k](https://github.com/openai/gym)

- [EUCLID: Towards Efficient Unsupervised Reinforcement Learning with Multi-choice Dynamics Model](https://openreview.net/forum?id=xQAjSr64PTc)
  - Yifu Yuan, Jianye HAO, Fei Ni, Yao Mu, YAN ZHENG, Yujing Hu, Jinyi Liu, Yingfeng Chen, Changjie Fan
  - Key: jointly pretrain the multi-headed dynamics model and unsupervised exploration policy, finetune to downstream tasks
  - OpenReview: 6, 6, 6, 6
  - ExpEnv: [URLB benchmark](https://github.com/rll-research/url_benchmark)

- [Choreographer: Learning and Adapting Skills in Imagination](https://openreview.net/forum?id=PhkWyijGi5b)
  - Pietro Mazzaglia, Tim Verbelen, Bart Dhoedt, Alexandre Lacoste, Sai Rajeswar
  - Key: world model, skill discovery, skill learning, Skill adaptation
  - OpenReview: 8, 8, 6, 6
  - ExpEnv: [deepmind control suite](https://github.com/deepmind/dm_control), [Meta-World](https://github.com/Farama-Foundation/Metaworld)

</details>

### NeurIPS 2022

<details open>
<summary>Toggle</summary>

- [Bidirectional Learning for Offline Infinite-width Model-based Optimization](https://openreview.net/forum?id=_j8yVIyp27Q)
  - Can Chen, Yingxue Zhang, Jie Fu, Xue Liu, Mark Coates
  - Key: model-based, offline
  - OpenReview: 7, 6, 5
  - ExpEnv: [design-bench](https://github.com/rail-berkeley/design-bench)

- [A Unified Framework for Alternating Offline Model Training and Policy Learning](https://openreview.net/forum?id=5yjM1sQ1uKZ)
  - Shentao Yang, Shujian Zhang, Yihao Feng, Mingyuan Zhou
  - Key: model-based, offline, marginal importance weight
  - OpenReview: 7, 6, 6, 5
  - ExpEnv: [d4rl dataset](https://github.com/rail-berkeley/d4rl)

- [Model-Based Offline Reinforcement Learning with Pessimism-Modulated Dynamics Belief](https://openreview.net/forum?id=oDWyVsHBzNT)
  - Kaiyang Guo, Shao Yunfeng, Yanhui Geng
  - Key: model-based, offline
  - OpenReview: 8, 8, 7, 7
  - ExpEnv: [d4rl dataset](https://github.com/rail-berkeley/d4rl)

- [Double Check Your State Before Trusting It: Confidence-Aware Bidirectional Offline Model-Based Imagination](https://openreview.net/forum?id=3e3IQMLDSLP)
  - Jiafei Lyu, Xiu Li, Zongqing Lu
  - Key: double check mechanism, bidirectional modeling, offline RL
  - OpenReview: 7, 6, 6
  - ExpEnv: [d4rl dataset](https://github.com/rail-berkeley/d4rl)

- [Model-Based Opponent Modeling](https://arxiv.org/abs/2108.01843)
  - XiaoPeng Yu, Jiechuan Jiang, Wanpeng Zhang, Haobin Jiang, Zongqing Lu
  - Key: multi-agent, model-based
  - OpenReview: 7, 6, 4, 3
  - ExpEnv: [mpe](https://github.com/openai/multiagent-particle-envs), [google research football](https://github.com/google-research/football)

- [Mingling Foresight with Imagination: Model-Based Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2204.09418)
  - Zhiwei Xu, Dapeng Li, Bin Zhang, Yuan Zhan, Yunpeng Bai, Guoliang Fan
  - Key: multi-agent, model-based
  - OpenReview: 6, 5
  - ExpEnv: [StarCraft II](https://github.com/deepmind/pysc2), [Google Research Football](https://github.com/google-research/football), [Multi-Agent Discrete MuJoCo](https://github.com/schroederdewitt/multiagent_mujoco)

- [MoCoDA: Model-based Counterfactual Data Augmentation](https://openreview.net/forum?id=w6tBOjPCrIO)
  - Silviu Pitis, Elliot Creager, Ajay Mandlekar, Animesh Garg
  - Key: data augmentation framework, offline RL
  - OpenReview: 7, 7, 7, 6
  - ExpEnv: [2D Navigation](https://github.com/spitis/mocoda/blob/main/augment_offline_toy.py#L45), [Hook-Sweep](https://github.com/spitis/mrl/blob/master/envs/customfetch/custom_fetch.py#L1699)

- [When to Update Your Model: Constrained Model-based Reinforcement Learning](https://openreview.net/forum?id=9a1oV7UunyP)
  - Tianying Ji, Yu Luo, Fuchun Sun, Mingxuan Jing, Fengxiang He, Wenbing Huang
  - Key: event-triggered mechanism, constrained model-shift lower-bound optimization
  - OpenReview: 6, 6, 5, 5
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py)

- [Model-based Safe Deep Reinforcement Learning via a Constrained Proximal Policy Optimization Algorithm](https://openreview.net/forum?id=hYa_lseXK8)
  - Ashish Jayant, Shalabh Bhatnagar
  - Key: constrained RL, model-based
  - OpenReview: 7, 6, 5, 5
  - ExpEnv: [safety gym](https://github.com/openai/safety-gym)

- [Learning to Attack Federated Learning: A Model-based Reinforcement Learning Attack Framework](https://openreview.net/forum?id=4OHRr7gmhd4)
  - Henger Li, Xiaolin Sun, Zizhan Zheng
  - Key: attack & defense,  federated learning, model-based
  - OpenReview: 6, 6, 6, 5
  - ExpEnv: MNIST, FashionMNIST, EMNIST, CIFAR-10 and synthetic dataset

- [Model-Based Imitation Learning for Urban Driving](https://openreview.net/forum?id=Zk1SbbdZwS)
  - Anthony Hu, Gianluca Corrado, Nicolas Griffiths, Zachary Murez, Corina Gurau, Hudson Yeo, Alex Kendall, Roberto Cipolla, Jamie Shotton
  - Key: model-based, imitation learning, autonomous driving
  - OpenReview: 7, 6, 6
  - ExpEnv: [CARLA](https://github.com/wayveai/mile/tree/main/carla_gym)

- [Data-Driven Model-Based Optimization via Invariant Representation Learning](https://openreview.net/forum?id=gKe_A-DxzkH)
  - Han Qi, Yi Su, Aviral Kumar, Sergey Levine
  - Key: domain adaptation, invariant objective models, representation learning (no about model-based RL)
  - OpenReview: 7, 6, 6, 5, 5
  - ExpEnv: [design-bench](https://github.com/rail-berkeley/design-bench)

- [Model-based Lifelong Reinforcement Learning with Bayesian Exploration](https://openreview.net/forum?id=6I3zJn9Slsb)
  - Haotian Fu, Shangqun Yu, Michael Littman, George Konidaris
  - Key: lifelong RL, variational bayesian
  - OpenReview: 7, 6, 6
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py), [meta-world](https://github.com/rlworkgroup/metaworld)

- [Plan To Predict: Learning an Uncertainty-Foreseeing Model For Model-Based Reinforcement Learning](https://openreview.net/forum?id=L9YayWPcHA_)
  - Zifan Wu, Chao Yu, Chen Chen, Jianye Hao, Hankz Hankui Zhuo
  - Key: treat the model rollout process as a sequential decision making problem
  - OpenReview: 7, 7, 6, 6
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py), [d4rl](https://github.com/rail-berkeley/d4rl)

- [Joint Model-Policy Optimization of a Lower Bound for Model-Based RL](https://openreview.net/forum?id=LYfFj-Vk6lt)
  - Benjamin Eysenbach, Alexander Khazatsky, Sergey Levine, Russ Salakhutdinov
  - Key: unified objective for model-based RL
  - OpenReview: 8, 8, 7, 6
  - ExpEnv: [gridworld](https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/gridworld.py), [mujoco](https://github.com/openai/mujoco-py), [ROBEL manipulation](https://github.com/google-research/robel)

- [RAMBO-RL: Robust Adversarial Model-Based Offline Reinforcement Learning](https://openreview.net/forum?id=nrksGSRT7kX)
  - Marc Rigter, Bruno Lacerda, Nick Hawes
  - Key: offline rl, model-based rl, two-player game, adversarial model training
  - OpenReview: 6, 6, 6, 4
  - ExpEnv: [d4rl](https://github.com/rail-berkeley/d4rl)

- [Conservative Dual Policy Optimization for Efficient Model-Based Reinforcement Learning](https://openreview.net/forum?id=xL7B5axplIe)
  - Shenao Zhang
  - Key: posterior sampling RL, referential update, constrained conservative update
  - OpenReview: 7, 7, 5, 5
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py), [N-Chain MDPs](https://github.com/stratisMarkou/sample-efficient-bayesian-rl/blob/master/code/Environments.py)

- [Bayesian Optimistic Optimization: Optimistic Exploration for Model-based Reinforcement Learning](https://openreview.net/forum?id=GdHVClGh9N)
  - Chenyang Wu, Tianci Li, Zongzhang Zhang, Yang Yu
  - Key: optimism in the face of uncertainty(OFU), BOO Regret
  - OpenReview: 6, 6, 5
  - ExpEnv: [RiverSwim, Chain, Random MDPs]()

- [Model-based RL with Optimistic Posterior Sampling: Structural Conditions and Sample Complexity](https://openreview.net/forum?id=bEMrmaw8gOB)
  - Alekh Agarwal, Tong Zhang
  - Key: posterior sampling RL, Bellman error decoupling framework
  - OpenReview: 7, 7, 7, 6
  - ExpEnv: None

- [Exponential Family Model-Based Reinforcement Learning via Score Matching](https://openreview.net/forum?id=G1uywu6vNZe)
  - Gene Li, Junbo Li, Nathan Srebro, Zhaoran Wang, Zhuoran Yang
  - Key: optimistic model-based, score matching
  - OpenReview: 7, 7, 6
  - ExpEnv: None

- [Deep Hierarchical Planning from Pixels](https://openreview.net/forum?id=wZk69kjy9_d)
  - Danijar Hafner, Kuang-Huei Lee, Ian Fischer, Pieter Abbeel
  - Key: hierarchical RL, long-horizon and sparse reward tasks
  - OpenReview: 6, 6, 5
  - ExpEnv: [atari](https://github.com/openai/gym), [deepmind control suite](https://github.com/deepmind/dm_control), [deepmind lab](https://github.com/deepmind/lab), [crafter](https://github.com/danijar/crafter)

- [Continuous MDP Homomorphisms and Homomorphic Policy Gradient](https://arxiv.org/abs/2209.07364)
  - Sahand Rezaei-Shoshtari, Rosie Zhao, Prakash Panangaden, David Meger, Doina Precup
  - Key: Homomorphic Policy Gradient, Continuous MDP Homomorphisms, Lax Bisimulation Loss
  - OpenReview: 7, 7, 7
  - ExpEnv: [deepmind control suite](https://github.com/deepmind/dm_control)

</details>

### ICML 2022

<details open>
<summary>Toggle</summary>

- [DreamerPro: Reconstruction-Free Model-Based Reinforcement Learning with Prototypical Representations](https://arxiv.org/abs/2110.14565)
  - Fei Deng, Ingook Jang, Sungjin Ahn
  - Key: dreamer, prototypes
  - ExpEnv: [deepmind control suite](https://github.com/deepmind/dm_control)

- [Denoised MDPs: Learning World Models Better Than the World Itself](https://arxiv.org/pdf/2206.15477.pdf)
  - Tongzhou Wang, Simon Du, Antonio Torralba, Phillip Isola, Amy Zhang, Yuandong Tian
  - Key: representation learning, denoised model
  - ExpEnv: [deepmind control suite](https://github.com/deepmind/dm_control), [RoboDesk](https://github.com/SsnL/robodesk)

- [Model-based Meta Reinforcement Learning using Graph Structured Surrogate Models and Amortized Policy Search](https://arxiv.org/pdf/2102.08291.pdf)
  - Qi Wang, Herke van Hoof
  - Key: graph structured surrogate model, meta training
  - ExpEnv: [atari, mujoco](https://github.com/openai/gym)

- [Towards Adaptive Model-Based Reinforcement Learning](https://arxiv.org/pdf/2204.11464.pdf)
  - Yi Wan, Ali Rahimi-Kalahroudi, Janarthanan Rajendran, Ida Momennejad, Sarath Chandar, Harm van Seijen
  - Key: local change adaptation
  - ExpEnv: [GridWorldLoCA, ReacherLoCA, MountaincarLoCA](https://github.com/chandar-lab/LoCA2)

- [Efficient Model-based Multi-agent Reinforcement Learning via Optimistic Equilibrium Computation](https://arxiv.org/pdf/2203.07322.pdf)
  - Pier Giuseppe Sessa, Maryam Kamgarpour, Andreas Krause
  - Key: model-based multi-agent, confidence bound
  - ExpEnv: [SMART](https://github.com/huawei-noah/SMARTS)

- [Regularizing a Model-based Policy Stationary Distribution to Stabilize Offline Reinforcement Learning](https://arxiv.org/pdf/2206.07166.pdf)
  - Shentao Yang, Yihao Feng, Shujian Zhang, Mingyuan Zhou
  - Key: offline rl, model-based rl, stationary distribution regularization
  - ExpEnv: [d4rl](https://github.com/rail-berkeley/d4rl)

- [Design-Bench: Benchmarks for Data-Driven Offline Model-Based Optimization](https://arxiv.org/pdf/2202.08450.pdf)
  - Brandon Trabucco, Xinyang Geng, Aviral Kumar, Sergey Levine
  - Key: benchmark, offline MBO
  - ExpEnv: [Design-Bench Benchmark Tasks](https://github.com/rail-berkeley/design-bench)

- [Temporal Difference Learning for Model Predictive Control](https://arxiv.org/pdf/2203.04955.pdf)
  - Nicklas Hansen, Hao Su, Xiaolong Wang
  - Key: td-learning, MPC
  - ExpEnv: [deepmind control suite](https://github.com/deepmind/dm_control), [Meta-World](https://github.com/rlworkgroup/metaworld)

</details>

### ICLR 2022

<details open>
<summary>Toggle</summary>

- [Revisiting Design Choices in Offline Model Based Reinforcement Learning](https://openreview.net/forum?id=zz9hXVhf40)
  - Cong Lu, Philip Ball, Jack Parker-Holder, Michael Osborne, Stephen J. Roberts
  - Key: model-based offline, uncertainty quantification
  - OpenReview: 8, 8, 6, 6, 6
  - ExpEnv: [d4rl dataset](https://github.com/rail-berkeley/d4rl)

- [Value Gradient weighted Model-Based Reinforcement Learning](https://openreview.net/forum?id=4-D6CZkRXxI)
  - Claas A Voelcker, Victor Liao, Animesh Garg, Amir-massoud Farahmand
  - Key: Value-Gradient weighted Model loss
  - OpenReview: 8, 8, 6, 6
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py)

- [Planning in Stochastic Environments with a Learned Model](https://openreview.net/forum?id=X6D9bAHhBQ1)
  - Ioannis Antonoglou, Julian Schrittwieser, Sherjil Ozair, Thomas K Hubert, David Silver
  - Key: MCTS, stochastic MuZero
  - OpenReview: 10, 8, 8, 5
  - ExpEnv: 2048 game, Backgammon, Go

- [Policy improvement by planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO)
  - Ivo Danihelka, Arthur Guez, Julian Schrittwieser, David Silver
  - Key: Gumbel AlphaZero, Gumbel MuZero
  - OpenReview: 8, 8, 8, 6
  - ExpEnv: go, chess, [atari](https://github.com/openai/gym)

- [Model-Based Offline Meta-Reinforcement Learning with Regularization](https://openreview.net/forum?id=EBn0uInJZWh)
  - Sen Lin, Jialin Wan, Tengyu Xu, Yingbin Liang, Junshan Zhang
  - Key: model-based offline Meta-RL
  - OpenReview: 8, 6, 6, 6
  - ExpEnv: [d4rl dataset](https://github.com/rail-berkeley/d4rl)

- [On-Policy Model Errors in Reinforcement Learning](https://openreview.net/forum?id=81e1aeOt-sd)
  - Lukas Froehlich, Maksym Lefarov, Melanie Zeilinger, Felix Berkenkamp
  - Key: model errors, on-policy corrections
  - OpenReview: 8, 6, 6, 5
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py), [pybullet](https://github.com/benelot/pybullet-gym)

- [A Relational Intervention Approach for Unsupervised Dynamics Generalization in Model-Based Reinforcement Learning](https://openreview.net/forum?id=YRq0ZUnzKoZ)
  - Jiaxian Guo, Mingming Gong, Dacheng Tao
  - Key: relational intervention, dynamics generalization
  - OpenReview: 8, 8, 6, 6
  - ExpEnv: [Pendulum](https://github.com/openai/gym), [mujoco](https://github.com/openai/mujoco-py)

- [Information Prioritization through Empowerment in Visual Model-based RL](https://openreview.net/forum?id=DfUjyyRW90)
  - Homanga Bharadhwaj, Mohammad Babaeizadeh, Dumitru Erhan, Sergey Levine
  - Key: mutual information, visual model-based RL
  - OpenReview: 8, 8, 8, 6
  - ExpEnv: [deepmind control suite](https://github.com/deepmind/dm_control), [Kinetics dataset](https://github.com/cvdfoundation/kinetics-dataset)

- [Transfer RL across Observation Feature Spaces via Model-Based Regularization](https://openreview.net/forum?id=7KdAoOsI81C)
  - Yanchao Sun, Ruijie Zheng, Xiyao Wang, Andrew E Cohen, Furong Huang
  - Key: latent dynamics model, transfer RL
  - OpenReview: 8, 6, 5, 5
  - ExpEnv: [CartPole, Acrobot and Cheetah-Run](https://github.com/openai/gym), [mujoco](https://github.com/openai/mujoco-py), [3DBall](https://github.com/Unity-Technologies/ml-agents)

- [Learning State Representations via Retracing in Reinforcement Learning](https://openreview.net/forum?id=CLpxpXqqBV)
  - Changmin Yu, Dong Li, Jianye HAO, Jun Wang, Neil Burgess
  - Key: representation learning, learning via retracing
  - OpenReview: 8, 6, 5, 3
  - ExpEnv: [deepmind control suite](https://github.com/deepmind/dm_control)

- [Model-augmented Prioritized Experience Replay](https://openreview.net/forum?id=WuEiafqdy9H)
  - Youngmin Oh, Jinwoo Shin, Eunho Yang, Sung Ju Hwang
  - Key: prioritized experience replay, mbrl
  - OpenReview: 8, 8, 6, 5
  - ExpEnv: [pybullet](https://github.com/benelot/pybullet-gym)

- [Evaluating Model-Based Planning and Planner Amortization for Continuous Control](https://openreview.net/forum?id=SS8F6tFX3-)
  - Arunkumar Byravan, Leonard Hasenclever, Piotr Trochim, Mehdi Mirza, Alessandro Davide Ialongo, Yuval Tassa, Jost Tobias Springenberg, Abbas Abdolmaleki, Nicolas Heess, Josh Merel, Martin Riedmiller
  - Key: model predictive control
  - OpenReview: 8, 6, 6, 6
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py)

- [Gradient Information Matters in Policy Optimization by Back-propagating through Model](https://openreview.net/forum?id=rzvOQrnclO0)
  - Chongchong Li, Yue Wang, Wei Chen, Yuting Liu, Zhi-Ming Ma, Tie-Yan Liu
  - Key: two-model-based method, analyze model error and policy gradient
  - OpenReview: 8, 8, 6, 6
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py)

- [Pareto Policy Pool for Model-based Offline Reinforcement Learning](https://openreview.net/forum?id=OqcZu8JIIzS)
  - Yijun Yang, Jing Jiang, Tianyi Zhou, Jie Ma, Yuhui Shi
  - Key: model-based offline, model return-uncertainty trade-off
  - OpenReview: 8, 8, 6, 5
  - ExpEnv: [d4rl dataset](https://github.com/rail-berkeley/d4rl)

- [Pessimistic Model-based Offline Reinforcement Learning under Partial Coverage](https://openreview.net/forum?id=tyrJsbKAe6)
  - Masatoshi Uehara, Wen Sun
  - Key: model-based offline theory, PAC bounds
  - OpenReview: 8, 6, 6, 5
  - ExpEnv: None

- [Know Thyself: Transferable Visual Control Policies Through Robot-Awareness](https://openreview.net/forum?id=o0ehFykKVtr)
  - Edward S. Hu, Kun Huang, Oleh Rybkin, Dinesh Jayaraman
  - Key: world models that transfer to new robots
  - OpenReview: 8, 6, 6, 5
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py), WidowX and Franka Panda robot

</details>

### NeurIPS 2021

<details open>
<summary>Toggle</summary>

- [On Effective Scheduling of Model-based Reinforcement Learning](https://arxiv.org/abs/2111.08550)
  - Hang Lai, Jian Shen, Weinan Zhang, Yimin Huang, Xing Zhang, Ruiming Tang, Yong Yu, Zhenguo Li
  - Key: extension of mbpo, hyper-controller learning
  - OpenReview: 8, 6, 6
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py), [pybullet](https://github.com/benelot/pybullet-gym)

- [COMBO: Conservative Offline Model-Based Policy Optimization](https://openreview.net/pdf?id=dUEpGV2mhf)
  - Tianhe Yu, Aviral Kumar, Rafael Rafailov, Aravind Rajeswaran, Sergey Levine, Chelsea Finn
  - Key: offline reinforcement learning, model-based reinforcement learning, deep reinforcement learning
  - OpenReview: 6, 7, 6, 8
  - ExpEnv: [d4rl dataset](https://github.com/rail-berkeley/d4rl)

- [Safe Reinforcement Learning by Imagining the Near Future](https://arxiv.org/abs/2202.07789)
  - Garrett Thomas, Yuping Luo, Tengyu Ma
  - Key: safe rl, reward penalty, theory about model-based rollouts
  - OpenReview: 8, 6, 6
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py)

- [Model-Based Reinforcement Learning via Imagination with Derived Memory](https://openreview.net/forum?id=jeATherHHGj)
  - Yao Mu, Yuzheng Zhuang, Bin Wang, Guangxiang Zhu, Wulong Liu, Jianyu Chen, Ping Luo, Shengbo Eben Li, Chongjie Zhang, Jianye HAO
  - Key: extension of dreamer, prediction-reliability weight
  - OpenReview: 6, 6, 6, 6
  - ExpEnv: [deepmind control suite](https://github.com/deepmind/dm_control)

- [MobILE: Model-Based Imitation Learning From Observation Alone](https://arxiv.org/abs/2102.10769)
  - Rahul Kidambi, Jonathan Chang, Wen Sun
  - Key: imitation learning from observations alone, mbrl
  - OpenReview: 6, 6, 6, 4
  - ExpEnv: [cartpole](https://github.com/openai/gym), [mujoco](https://github.com/openai/mujoco-py)

- [Model-Based Episodic Memory Induces Dynamic Hybrid Controls](https://arxiv.org/abs/2111.02104)
  - Hung Le, Thommen Karimpanal George, Majid Abdolshah, Truyen Tran, Svetha Venkatesh
  - Key: model-based, episodic control
  - OpenReview: 7, 7, 6, 6
  - ExpEnv: [2D maze navigation](https://github.com/MattChanTK/gym-maze), [cartpole, mountainCar and lunarlander](https://github.com/openai/gym), [atari](https://gym.openai.com/envs/atari), [3D navigation: gym-miniworld](https://github.com/maximecb/gym-miniworld)

- [A Consciousness-Inspired Planning Agent for Model-Based Reinforcement Learning](https://arxiv.org/abs/2106.02097)
  - Mingde Zhao, Zhen Liu, Sitao Luan, Shuyuan Zhang, Doina Precup, Yoshua Bengio
  - Key: mbrl, set representation
  - OpenReview: 7, 7, 7, 6
  - ExpEnv: [MiniGrid-BabyAI framework](https://github.com/maximecb/gym-minigrid)

- [Mastering Atari Games with Limited Data](https://openreview.net/forum?id=OKrNPg3xR3T)
  - Weirui Ye, Shaohuai Liu, Thanard Kurutach, Pieter Abbeel, Yang Gao
  - Key: muzero, self-supervised consistency loss
  - OpenReview: 7, 7, 7, 5
  - ExpEnv: [atrai 100k](https://github.com/openai/gym), [deepmind control suite](https://github.com/deepmind/dm_control)

- [Online and Offline Reinforcement Learning by Planning with a Learned Model](https://openreview.net/forum?id=HKtsGW-lNbw)
  - Julian Schrittwieser, Thomas K Hubert, Amol Mandhane, Mohammadamin Barekatain, Ioannis Antonoglou, David Silver
  - Key: muzero, reanalyse, offline
  - OpenReview: 8, 8, 7, 6
  - ExpEnv: [atrai dataset, deepmind control suite dataset](https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged)

- [Self-Consistent Models and Values](https://arxiv.org/abs/2110.12840)
  - Gregory Farquhar, Kate Baumli, Zita Marinho, Angelos Filos, Matteo Hessel, Hado van Hasselt, David Silver
  - Key: new model learning way
  - OpenReview: 7, 7, 7, 6
  - ExpEnv: tabular MDP, Sokoban, [atari](https://github.com/openai/gym)

- [Proper Value Equivalence](https://arxiv.org/abs/2106.10316)
  - Christopher Grimm, Andre Barreto, Gregory Farquhar, David Silver, Satinder Singh
  - Key: value equivalence, value-based planning, muzero
  - OpenReview: 8, 7, 7, 6
  - ExpEnv: [four rooms](https://github.com/maximecb/gym-minigrid), [atari](https://github.com/openai/gym)

- [MOPO: Model-based Offline Policy Optimization](https://arxiv.org/abs/2005.13239)
  - Tianhe Yu, Garrett Thomas, Lantao Yu, Stefano Ermon, James Zou, Sergey Levine, Chelsea Finn, Tengyu Ma
  - Key: model-based, offline
  - OpenReview: None
  - ExpEnv: [d4rl dataset](https://github.com/rail-berkeley/d4rl), halfcheetah-jump and ant-angle

- [RoMA: Robust Model Adaptation for Offline Model-based Optimization](https://arxiv.org/abs/2110.14188)
  - Sihyun Yu, Sungsoo Ahn, Le Song, Jinwoo Shin
  - Key: model-based, offline
  - OpenReview: 7, 6, 6
  - ExpEnv: [design-bench](https://github.com/brandontrabucco/design-bench)

- [Offline Reinforcement Learning with Reverse Model-based Imagination](https://arxiv.org/abs/2110.00188)
  - Jianhao Wang, Wenzhe Li, Haozhe Jiang, Guangxiang Zhu, Siyuan Li, Chongjie Zhang
  - Key: model-based, offline
  - OpenReview: 7, 6, 6, 5
  - ExpEnv: [d4rl dataset](https://github.com/rail-berkeley/d4rl)

- [Offline Model-based Adaptable Policy Learning](https://openreview.net/forum?id=lrdXc17jm6)
  - Xiong-Hui Chen, Yang Yu, Qingyang Li, Fan-Ming Luo, Zhiwei Tony Qin, Shang Wenjie, Jieping Ye
  - Key: model-based, offline
  - OpenReview: 6, 6, 6, 4
  - ExpEnv: [d4rl dataset](https://github.com/rail-berkeley/d4rl)

- [Weighted model estimation for offline model-based reinforcement learning](https://openreview.net/pdf?id=zdC5eXljMPy)
  - Toru Hishinuma, Kei Senda
  - Key: model-based, offline, off-policy evaluation
  - OpenReview: 7, 6, 6, 6
  - ExpEnv: pendulum, [d4rl dataset](https://github.com/rail-berkeley/d4rl)

- [Reward-Free Model-Based Reinforcement Learning with Linear Function Approximation](https://arxiv.org/abs/2110.06394)
  - Weitong Zhang, Dongruo Zhou, Quanquan Gu
  - Key: learning theory, model-based reward-free RL, linear function approximation
  - OpenReview: 6, 6, 5, 5
  - ExpEnv: None

- [Provable Model-based Nonlinear Bandit and Reinforcement Learning: Shelve Optimism, Embrace Virtual Curvature](https://arxiv.org/abs/2102.04168)
  - Kefan Dong, Jiaqi Yang, Tengyu Ma
  - Key: learning theory, model-based bandit RL, nonlinear function approximation
  - OpenReview: 7, 7, 7, 6
  - ExpEnv: None

- [Discovering and Achieving Goals via World Models](https://openreview.net/forum?id=6vWuYzkp8d)
  - Russell Mendonca, Oleh Rybkin, Kostas Daniilidis, Danijar Hafner, Deepak Pathak
  - Key: unsupervised goal reaching, goal-conditioned RL
  - OpenReview: 6, 6, 6, 6, 6
  - ExpEnv: [walker, quadruped, bins, kitchen](https://github.com/orybkin/lexa-benchmark)

</details>

### ICLR 2021

<details open>
<summary>Toggle</summary>

- [Deployment-Efficient Reinforcement Learning via Model-Based Offline Optimization](https://arxiv.org/abs/2006.03647)
  - Tatsuya Matsushima, Hiroki Furuta, Yutaka Matsuo, Ofir Nachum, Shixiang Gu
  - Key: model-based, behavior cloning (warmup), trpo
  - OpenReview: 8, 7, 7, 5
  - ExpEnv: [d4rl dataset](https://github.com/rail-berkeley/d4rl)

- [Control-Aware Representations for Model-based Reinforcement Learning](https://arxiv.org/abs/2006.13408)
  - Brandon Cui, Yinlam Chow, Mohammad Ghavamzadeh
  - Key: representation learning, model-based soft actor-critic
  - OpenReview: 6, 6, 6
  - ExpEnv: planar system, inverted pendulum – swingup, cartpole, 3-link manipulator — swingUp & balance

- [Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193)
  - Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, Jimmy Ba
  - Key: DreamerV2, many tricks(multiple categorical variables, KL balancing, etc)
  - OpenReview: 9, 8, 5, 4
  - ExpEnv: [atari](https://github.com/openai/gym)

- [Model-Based Visual Planning with Self-Supervised Functional Distances](https://openreview.net/forum?id=UcoXdfrORC)
  - Stephen Tian, Suraj Nair, Frederik Ebert, Sudeep Dasari, Benjamin Eysenbach, Chelsea Finn, Sergey Levine
  - Key: goal-reaching task, dynamics learning, distance learning (goal-conditioned Q-function)
  - OpenReview: 7, 7, 7, 7
  - ExpEnv: [sawyer](https://github.com/rlworkgroup/metaworld/tree/master/metaworld/envs), door sliding

- [Model-Based Offline Planning](https://arxiv.org/abs/2008.05556)
  - Arthur Argenson, Gabriel Dulac-Arnold
  - Key: model-based, offline
  - OpenReview: 8, 7, 5, 5
  - ExpEnv: [RL Unplugged(RLU)](https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged), [d4rl dataset](https://github.com/rail-berkeley/d4rl)

- [Offline Model-Based Optimization via Normalized Maximum Likelihood Estimation](https://arxiv.org/abs/2102.07970)
  - Justin Fu, Sergey Levine
  - Key: model-based, offline
  - OpenReview: 8, 6, 6
  - ExpEnv: [design-bench](https://github.com/brandontrabucco/design-bench)

- [On the role of planning in model-based deep reinforcement learning](https://arxiv.org/abs/2011.04021)
  - Jessica B. Hamrick, Abram L. Friesen, Feryal Behbahani, Arthur Guez, Fabio Viola, Sims Witherspoon, Thomas Anthony, Lars Buesing, Petar Veličković, Théophane Weber
  - Key: discussion about planning in MuZero
  - OpenReview: 7, 7, 6, 5
  - ExpEnv: [atari](https://github.com/openai/gym), go, [deepmind control suite](https://github.com/deepmind/dm_control)

- [Representation Balancing Offline Model-based Reinforcement Learning](https://openreview.net/forum?id=QpNz8r_Ri2Y)
  - Byung-Jun Lee, Jongmin Lee, Kee-Eung Kim
  - Key: Representation Balancing MDP, model-based, offline
  - OpenReview: 7, 7, 7, 6
  - ExpEnv: [d4rl dataset](https://github.com/rail-berkeley/d4rl)

- [Model-based micro-data reinforcement learning: what are the crucial model properties and which model to choose?](https://openreview.net/forum?id=p5uylG94S68)
  - Balázs Kégl, Gabriel Hurtado, Albert Thomas
  - Key: mixture density nets, heteroscedasticity
  - OpenReview: 7, 7, 7, 6, 5
  - ExpEnv: [acrobot system](https://github.com/openai/gym)

</details>

### ICML 2021

<details open>
<summary>Toggle</summary>

- [Conservative Objective Models for Effective Offline Model-Based Optimization](https://arxiv.org/abs/2107.06882)
  - Brandon Trabucco, Aviral Kumar, Xinyang Geng, Sergey Levine
  - Key: conservative objective model, offline mbrl
  - ExpEnv: [design-bench](https://github.com/brandontrabucco/design-bench)

- [Continuous-Time Model-Based Reinforcement Learning](https://arxiv.org/abs/2102.04764)
  - Çağatay Yıldız, Markus Heinonen, Harri Lähdesmäki
  - Key: continuous-time
  - ExpEnv: [pendulum, cartPole and acrobot](https://github.com/openai/gym)

- [Model-Based Reinforcement Learning via Latent-Space Collocation](https://arxiv.org/abs/2106.13229)
  - Oleh Rybkin, Chuning Zhu, Anusha Nagabandi, Kostas Daniilidis, Igor Mordatch, Sergey Levine
  - Key: latent space collocation
  - ExpEnv: [sparse metaworld tasks](https://github.com/rlworkgroup/metaworld/tree/master/metaworld/envs)

- [Model-Free and Model-Based Policy Evaluation when Causality is Uncertain](http://proceedings.mlr.press/v139/bruns-smith21a.html)
  - David A Bruns-Smith
  - Key: worst-case bounds
  - ExpEnv: [ope-tools](https://github.com/clvoloshin/COBS)

- [Muesli: Combining Improvements in Policy Optimization](https://arxiv.org/abs/2104.06159)
  - Matteo Hessel, Ivo Danihelka, Fabio Viola, Arthur Guez, Simon Schmitt, Laurent Sifre, Theophane Weber, David Silver, Hado van Hasselt
  - Key: value equivalence
  - ExpEnv: [atari](https://github.com/openai/gym)

- [Vector Quantized Models for Planning](https://arxiv.org/pdf/2106.04615.pdf)
  - Sherjil Ozair, Yazhe Li, Ali Razavi, Ioannis Antonoglou, Aäron van den Oord, Oriol Vinyals
  - Key: VQVAE, MCTS
  - ExpEnv: [chess datasets](https://www.ﬁcsgames.org/download.html), [DeepMind Lab](https://github.com/deepmind/lab)

- [PC-MLP: Model-based Reinforcement Learning with Policy Cover Guided Exploration](https://arxiv.org/abs/2107.07410)
  - Yuda Song, Wen Sun
  - Key: sample complexity, kernelized nonlinear regulators, linear MDPs
  - ExpEnv: [mountain car, antmaze](https://github.com/openai/gym), [mujoco](https://github.com/openai/mujoco-py)

- [Temporal Predictive Coding For Model-Based Planning In Latent Space](https://arxiv.org/abs/2106.07156)
  - Tung Nguyen, Rui Shu, Tuan Pham, Hung Bui, Stefano Ermon
  - Key: temporal predictive coding with a RSSM, latent space
  - ExpEnv: [deepmind control suite](https://github.com/deepmind/dm_control)

- [Model-based Reinforcement Learning for Continuous Control with Posterior Sampling](https://arxiv.org/abs/2012.09613)
  - Ying Fan, Yifei Ming
  - Key: regret bound of psrl, mpc
  - ExpEnv: [continuous cartpole, pendulum swingup](https://github.com/openai/gym), [mujoco](https://github.com/openai/mujoco-py)

- [A Sharp Analysis of Model-based Reinforcement Learning with Self-Play](https://arxiv.org/abs/2010.01604)
  - Qinghua Liu, Tiancheng Yu, Yu Bai, Chi Jin
  - Key: learning theory, multi-agent, model-based self play, two-player zero-sum Markov games
  - ExpEnv: None

</details>

### Other

- [UniZero: Generalized and Efficient Planning with Scalable Latent World Models](https://openreview.net/forum?id=Gl6dF9soQo) 
  - Yuan Pu, Yazhe Niu, Zhenjie Yang, Jiyuan Ren, Hongsheng Li, Yu Liu *TMLR2025*
  - Key: world model, MCTS, model-based reinforcement learning, transformer, latent planning, multitask learning  
  - ExpEnv: Atari, DMControl, VisualMatch

- [Driving into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Driving_into_the_Future_Multiview_Visual_Forecasting_and_Planning_with_CVPR_2024_paper.html)
  - Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, Zhaoxiang Zhang *CVPR 2024*
  - Key: AutoDrive world modeling
  - ExpEnv: [nuScenes]()

- [DriveWorld: 4D Pre-trained Scene Understanding via World Models for Autonomous Driving](https://openreview.net/pdf?id=tT3LUdmzbd)
  - Chen Min, Dawei Zhao, Liang Xiao, Jian Zhao, Xinli Xu, Zheng Zhu, Lei Jin, Jianshu Li, Yulan Guo, Junliang Xing, Liping Jing, Yiming Nie, Bin Dai *CVPR 2024*
  - Key: AutoDrive world modeling
  - ExpEnv: [nuScenes](), [OpenScene]()

- [Masked Trajectory Models for Prediction, Representation, and Control](https://openreview.net/pdf?id=tT3LUdmzbd)
  - Philipp Wu, Arjun Majumdar, Kevin Stone, Yixin Lin, Igor Mordatch, Pieter Abbeel, Aravind Rajeswaran *ICLR 2023 Workshop RRL*
  - Key: offline RL, learning for control, sequence modeling
  - ExpEnv: [d4rl](https://github.com/rail-berkeley/d4rl)

- [World Models via Policy-Guided Trajectory Diffusion](https://arxiv.org/abs/2312.08533)
  - Marc Rigter, Jun Yamada, Ingmar Posner *Arxiv 2023*
  - Key: Diffusion model, world model
  - ExpEnv: [deepmind control suite](https://github.com/deepmind/dm_control), [gridworld](https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/gridworld.py)

- [Model-Based Epistemic Variance of Values for Risk-Aware Policy Optimization](https://arxiv.org/abs/2312.04386)
  - Carlos E. Luis, Alessandro G. Bottero, Julia Vinogradska, Felix Berkenkamp, Jan Peters *Arxiv 2023*
  - Key: cumulative rewards uncertainty estimation in MBRL
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py)

- [Sample-Efficient Learning to Solve a Real-World Labyrinth Game Using Data-Augmented Model-Based Reinforcement Learning](https://arxiv.org/abs/2312.09906)
  - Thomas Bi, Raffaello D'Andrea. *Arxiv 2023*
  - Key: Data-Augmented,  DreamerV3
  - ExpEnv: [Real-World Labyrinth Game]()

- [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)
  - Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap. *Arxiv 2023*
  - Key: DreamerV3, scaling property to world model
  - ExpEnv: [deepmind control suite](https://github.com/deepmind/dm_control), [atari](https://github.com/openai/gym), [DMLab](https://github.com/deepmind/lab), [minecraft](https://github.com/minerllabs/minerl)

- [Theoretically Guaranteed Policy Improvement Distilled from Model-Based Planning](https://arxiv.org/abs/2307.12933)
  - Chuming Li, Ruonan Jia, Jiawei Yao, Jie Liu, Yinmin Zhang, Yazhe Niu, Yaodong Yang, Yu Liu, Wanli Ouyang. *IJCAI Workshop 2023*
  - Key: extended policy improvement, model regularization, planning theorem
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py)


## Tutorial

- [Video] [Csaba Szepesvári - The challenges of model-based reinforcement learning and how to overcome them](https://www.youtube.com/watch?v=-Y-fHsPIQ_Q)
- [Blog] [Model-Based Reinforcement Learning: Theory and Practice](https://bair.berkeley.edu/blog/2019/12/12/mbpo/)


## Codebase

- [mbrl-lib](https://github.com/facebookresearch/mbrl-lib) - Meta: Library for Model Based RL
- [DI-engine](https://github.com/opendilab/DI-engine) - OpenDILab: Decision AI Engine


## Contributing

Our purpose is to make this repo even better. If you are interested in contributing, please refer to [HERE](CONTRIBUTING.md) for instructions in contribution.


## License

Awesome Model-Based RL is released under the Apache 2.0 license.

<p align="right">(<a href="#top">Back to top</a>)</p>

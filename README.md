# Awesome Model-Based Reinforcement Learning

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://github.com/opendilab/awesome-model-based-RL)
![GitHub stars](https://img.shields.io/github/stars/opendilab/awesome-model-based-RL?color=yellow)
![GitHub forks](https://img.shields.io/github/forks/opendilab/awesome-model-based-RL?color=9cf)
[![GitHub license](https://img.shields.io/github/license/opendilab/awesome-model-based-RL)](https://github.com/opendilab/awesome-model-based-RL/blob/main/LICENSE)
<!-- ![visitors](https://visitor-badge.glitch.me/badge?page_id=opendilab/awesome-model-based-RL) -->

This is a collection of research papers for **model-based reinforcement learning (mbrl)**.
And the repository will be continuously updated to track the frontier of model-based rl.

Welcome to follow and star!

<pre name="code" class="html">
<font color="red">[2023.11.29] <b>New: We update the ICLR 2024 submissions of model-based rl!</b></font>

[2023.09.29] We update the NeurIPS 2023 paper list of model-based rl.

[2023.06.15] We update the ICML 2023 paper list of model-based rl!

[2023.02.05] We update the ICLR 2023 paper list of model-based rl.

[2022.11.03] We update the NeurIPS 2022 paper list of model-based rl.

[2022.07.06] We update the ICML 2022 paper list of model-based rl.

[2022.02.13] We update the ICLR 2022 paper list of model-based rl.

[2021.12.28] We release the awesome model-based rl.
</pre>


## Table of Contents

- [A Taxonomy of Model-Based RL Algorithms](#a-taxonomy-of-model-based-rl-algorithms)
- [Papers](#papers)
  - [Classic Model-Based RL Papers](#classic-model-based-rl-papers)
  - [ICLR 2024](#iclr-2024)(**<font color="red">New!!!</font>**)
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

### ICLR 2024

<details open>
<summary>Toggle</summary>

- [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://openreview.net/forum?id=Oxh5CstDJU)
  <!-- -  -->
  - Key: implicit world model, model predictive control, generalist td-mpc2
  - OpenReview: 8, 8, 8, 8
  - ExpEnv: [deepmind control suite](https://github.com/deepmind/dm_control), [Meta-World](https://github.com/Farama-Foundation/Metaworld), [maniskill2](https://github.com/haosulab/ManiSkill2), [myosuite](https://github.com/MyoHub/myosuite)

- [Robust Model Based Reinforcement Learning Using L1 Adaptive Control](https://openreview.net/forum?id=GaLCLvJaoF)
  <!-- -  -->
  - Key: L1 Adaptive Control
  - OpenReview: 8, 6, 6, 6
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py)

- [Mastering Memory Tasks with World Models](https://openreview.net/forum?id=1vDArHJ68h)
  <!-- -  -->
  - Key: recall to imagine module, based on DreamerV3
  - OpenReview: 10, 8, 6
  - ExpEnv: [bsuite](https://github.com/google-deepmind/bsuite), [popgym](https://github.com/proroklab/popgym), [atari](https://github.com/openai/gym), [deepmind control suite](https://github.com/deepmind/dm_control), [memory maze](https://github.com/jurgisp/memory-maze)

- [Learning Hierarchical World Models with Adaptive Temporal Abstractions from Discrete Latent Dynamics](https://openreview.net/forum?id=TjCDNssXKU)
  <!-- -  -->
  - Key: Context-specific Recurrent State Space Model, hierarchical world model
  - OpenReview: 8, 6, 6
  - ExpEnv: [MiniHack](https://github.com/facebookresearch/minihack), [VisualPinPad](https://github.com/danijar/director/blob/main/embodied/envs/pinpad.py), [MultiWorld](https://github.com/vitchyr/multiworld)

- [COPlanner: Plan to Roll Out Conservatively but to Explore Optimistically for Model-Based RL](https://openreview.net/forum?id=jnFcKjtUPN)
  <!-- -  -->
  - Key: conservative model rollouts, optimistic environment exploration
  - OpenReview: 6, 6, 6
  - ExpEnv: [mujoco](https://github.com/openai/mujoco-py), [deepmind control suite](https://github.com/deepmind/dm_control)

- [Efficient Multi-agent Reinforcement Learning by Planning](https://openreview.net/forum?id=CpnKq3UJwp)
  <!-- -  -->
  - Key: mcts, optimistic search lambda, advantage-weighted policy optimization
  - OpenReview: 8, 6, 6, 6
  - ExpEnv: [smac](https://github.com/oxwhirl/smac)

- [Differentiable Trajectory Optimization as a Policy Class for Reinforcement and Imitation Learning](https://openreview.net/forum?id=HL5P4H8eO2)
  <!-- -  -->
  - Key: differentiable trajectory optimization
  - OpenReview: 10, 8, 8, 5
  - ExpEnv: [deepmind control suite](https://github.com/deepmind/dm_control), [robomimic](https://github.com/ARISE-Initiative/robomimic), [maniskill](https://github.com/haosulab/ManiSkill2)

- [DMBP: Diffusion model based predictor for robust offline reinforcement learning against state observation perturbations](https://openreview.net/forum?id=ZULjcYLWKe)
  <!-- -  -->
  - Key: conditional diffusion, offline RL
  - OpenReview: 8, 8, 6, 6
  - ExpEnv: [d4rl](https://github.com/rail-berkeley/d4rl)

- [MAMBA: an Effective World Model Approach for Meta-Reinforcement Learning](https://openreview.net/forum?id=1RE0H6mU7M)
  <!-- -  -->
  - Key: context-based meta-RL, based on dreamer
  - OpenReview: 6, 6, 6, 6
  - ExpEnv: [Point Robot Navigation, Escape Room](https://github.com/Rondorf/BOReL/blob/main/environments/toy_navigation/point_robot.py), [Reacher Sparse](https://github.com/deepmind/dm_control)

- [Reward-Consistent Dynamics Models are Strongly Generalizable for Offline Reinforcement Learning](https://openreview.net/forum?id=GSBHKiw19c)
  <!-- -  -->
  - Key: reward learning, offline RL
  - OpenReview: 8, 6, 6, 6
  - ExpEnv: [d4rl](https://github.com/rail-berkeley/d4rl), [NeoRL](https://github.com/polixir/NeoRL)

- [DreamSmooth: Improving Model-based Reinforcement Learning via Reward Smoothing](https://openreview.net/forum?id=GruDNzQ4ux)
  <!-- -  -->
  - Key: learn to predict a temporally-smoothed reward rather than the exact reward at each timestep
  - OpenReview: 6, 6, 6, 5
  - ExpEnv: [robodesk](https://github.com/google-research/robodesk), [hand](https://github.com/openai/gym), [earthmoving](https://www.algoryx.se/agx-dynamics/)

- [Informed POMDP: Leveraging Additional Information in Model-Based RL](https://openreview.net/forum?id=5NJzNAXAmx)
  <!-- -  -->
  - Key: informed world model, based on DreamerV3
  - OpenReview: 6, 6, 6, 5
  - ExpEnv: [varying mountain hike](https://github.com/maximilianigl/DVRL/tree/master), [deepmind control suite](https://github.com/deepmind/dm_control), [pop gym](https://github.com/proroklab/popgym), [flickering atari and flickering control](https://github.com/openai/gym)

</details>

### NeurIPS 2023

<details open>
<summary>Toggle</summary>

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

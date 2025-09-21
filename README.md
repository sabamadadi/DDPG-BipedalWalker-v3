# Training a Reinforcement Learning Agent in BipedalWalker-v3

## Abstract  
In this work, we implement and train a reinforcement learning agent using the **Deep Deterministic Policy Gradient (DDPG)** algorithm in the **BipedalWalker-v3** environment from OpenAI Gym. The task requires controlling a bipedal robot with continuous action spaces to walk efficiently across rugged terrain. We design **Actor–Critic neural networks**, employ **experience replay** and **Ornstein–Uhlenbeck exploration noise**, and track training performance across 1000 episodes. Due to computational constraints, full convergence was not achieved, but the results demonstrate the ability of the DDPG agent to learn stable walking strategies.  

---

## 1. Introduction  
The BipedalWalker-v3 environment provides a challenging benchmark for reinforcement learning. The agent must control a robot with **24-dimensional continuous state space** and **4-dimensional continuous action space** to walk across uneven terrain. This makes the task highly non-trivial, as the agent must learn a sequence of continuous motor commands that maintain balance and forward motion.  

Traditional RL methods struggle with such continuous control problems. Policy-gradient based methods such as **Proximal Policy Optimization (PPO)**, **Deep Deterministic Policy Gradient (DDPG)**, and **Twin Delayed DDPG (TD3)** have been shown to perform better. In this project, we focus on implementing DDPG from scratch in **PyTorch** and applying it to BipedalWalker-v3.  

---

## 2. Methodology  

### 2.1 Algorithm Selection  
We implemented the **DDPG algorithm**, which combines **policy gradient** and **Q-learning** ideas. DDPG is an **off-policy, Actor–Critic method** designed for continuous action spaces.  

### 2.2 Neural Network Architectures  
Two deep networks are employed:  

- **Actor Network**  
  - Inputs: state vector (24 dimensions).  
  - Architecture: fully connected layers with Batch Normalization and ReLU activations.  
  - Output: action vector (4 dimensions) using `tanh` activation to keep actions within [-1, 1].  

- **Critic Network**  
  - Inputs: state and action pair.  
  - Architecture: fully connected layers, combining state and action in intermediate layers.  
  - Output: scalar Q-value estimating the value of a state–action pair.  

These networks are updated with **soft updates** of target networks using parameter τ = 0.001.  

### 2.3 Training Setup  
- **Episodes:** 1000  
- **Steps per episode:** up to 2000  
- **Replay buffer size:** 1,000,000  
- **Batch size:** 100  
- **Discount factor (γ):** 0.99  
- **Learning rates:** 1e-4 (Actor), 1e-3 (Critic)  
- **Exploration:** Ornstein–Uhlenbeck noise process  

### 2.4 Visualization  
A helper function `show_state()` was developed to visualize the agent’s interaction with the environment in real-time, providing a graphical display of the walker during training.  

---

## 3. Results  

Training was carried out for 1000 episodes. Despite limited compute resources (training was performed on a laptop, without access to a dedicated server or GPU cluster), the agent demonstrated the ability to improve its policy.  

Final logged metrics after 1000 episodes:  

- **Mean Score:** -47.17  
- **Mean Distance:** 39.83  
- **Actor Loss:** -4.07  
- **Critic Loss:** 1.76  

### Training Performance Visualization  

<p align="center">
  <img src="histo.png" alt="Training Reward Curve" width="600"/>
</p>  

---

## 4. Discussion  
The results indicate that the DDPG agent was able to partially solve the task, showing improvement in walking strategies compared to random behavior. However, due to computational limitations, the model was not trained to full convergence. More stable performance could likely be achieved with:  

- Longer training time on a GPU or server.  
- Hyperparameter tuning (batch size, learning rates, τ).  
- Reward shaping and curriculum learning for smoother exploration.  
- Considering TD3 (Twin Delayed DDPG) for improved stability.  

---

## 5. Conclusion  
This project demonstrates the successful implementation of a DDPG agent in PyTorch for the BipedalWalker-v3 environment. While the final performance was constrained by computational resources, the agent learned meaningful walking behaviors. This work highlights the challenges of continuous control in reinforcement learning and sets a foundation for future improvements using advanced techniques such as PPO or TD3.  

---

## References  
1. Lillicrap, T.P., et al. "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971 (2015).  
2. OpenAI Gym: https://gym.openai.com  
3. Sutton, R. S., & Barto, A. G. "Reinforcement Learning: An Introduction." MIT Press (2018).  

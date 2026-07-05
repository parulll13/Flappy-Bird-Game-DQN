# Flappy Bird AI using Deep Q-Network (DQN)

## Project Overview

This project implements a **Deep Q-Network (DQN)** agent that learns to play the **Flappy Bird** game using **Deep Reinforcement Learning**.

Instead of storing a huge Q-table like traditional Q-learning, a neural network is used to approximate Q-values.

The agent learns through:

* Experience Replay
* Epsilon-Greedy Policy
* Target Network
* Neural Network based Q-value prediction

---

## Project Structure

```text
FLAPPY BIRD PROJECT
│
├── runs/
│      └── Stores trained model (.pt) files and logs
│
├── agent.py
│      └── Main file for training/testing DQN agent
│
├── dqn.py
│      └── Defines Deep Q-Network architecture
│
├── experience_replay.py
│      └── Stores and samples experiences
│
├── parameters.yaml
│      └── Stores hyperparameters
│
└── __pycache__/
       └── Python cache files
```

---

## Technologies Used

* Python
* PyTorch
* Gymnasium
* Flappy Bird Gymnasium
* YAML

---

## Deep Q-Learning Workflow

```text
Environment
      ↓
Current State
      ↓
Policy Network
      ↓
Choose Action
      ↓
Perform Action
      ↓
Get Reward + Next State
      ↓
Store Experience
      ↓
Experience Replay
      ↓
Train Neural Network
      ↓
Update Target Network
```

---

## DQN Architecture

Neural Network structure:

```text
Input Layer (state_dim = 12)
          ↓
Hidden Layer (256 neurons)
          ↓
ReLU Activation
          ↓
Output Layer (action_dim = 2)
```

Actions:

```text
0 → Do Nothing
1 → Flap
```

---

## Hyperparameters

Hyperparameters are stored in:

```text
parameters.yaml
```

Current values:

```yaml
flappybirdv0:
    env_id: FlappyBird-v0
    epsilon_init: 1
    epsilon_min: 0.05
    epsilon_decay: 0.9995
    replay_memory_size: 100000
    mini_batch_size: 32
    network_sync_rate: 10
    alpha: 0.001
    gamma: 0.99
    reward_threshold: 1000
```

---

## Required Libraries

Install required libraries:

```bash
pip install torch
pip install gymnasium
pip install pyyaml
pip install flappy-bird-gymnasium
```

or:

```bash
pip install torch gymnasium pyyaml flappy-bird-gymnasium
```

---

## Create Environment (Recommended)

Create environment:

```bash
conda create -n flappy python=3.11
```

Activate environment:

```bash
conda activate flappy
```

---

## Training the Model

Run:

```bash
python agent.py flappybirdv0 --train
```

Training process:

```text
State
   ↓
Choose action
   ↓
Store experience
   ↓
Sample mini-batch
   ↓
Train network
   ↓
Update target network
```

---

## Testing Trained Model

Run:

```bash
python agent.py flappybirdv0
```

This loads the saved model from:

```text
runs/flappybirdv0.pt
```

and displays the trained Flappy Bird agent.

---

## Saved Files

During training:

```text
runs/
    flappybirdv0.pt
    flappybirdv0.log
```

### flappybirdv0.pt

Contains:

* trained neural network weights
* learned policy

### flappybirdv0.log

Contains:

* best rewards
* training progress

---

## Restarting Project After Closing Laptop

Open Anaconda Prompt

Activate environment:

```bash
conda activate flappy
```

Go to project folder:

```bash
cd "C:\Users\acer\Desktop\Flappy Bird Project"
```

Run training:

```bash
python agent.py flappybirdv0 --train
```

OR run testing:

```bash
python agent.py flappybirdv0
```

---

## Future Improvements

Possible improvements:

* Double DQN
* Dueling DQN
* Prioritized Experience Replay
* Reward shaping
* Model checkpointing
* Hyperparameter tuning

---

## Author

Parul Gupta

Deep Reinforcement Learning Mini Project

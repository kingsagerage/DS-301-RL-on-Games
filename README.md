# DS-301-RL-on-Games
DS-301 final project on reinforcement learning on tactic shooting simulation


## Description
The project aims to train double agents in deep reinforcement learning with DQN to perform tactical shooting tasks in 2D and 3D simulation environments built with OpenAI Gym.

## Getting Started

### Dependency
  * python >= 3.9
  * pip
  * numpy
  * pyparsing
  * jupyterlab
  * scikit-image
  * nvidia cudatoolkit=11.3
  * pytorch
  * torchvision
  * torchaudio
  * OpenAI gym

### Project Structure
This Project has 2 main parts:
- The train folder to train DQN agents in our 3D and 2D environments:
    - Within `train/Env/env.py` is where all the environment classes are stored
    - Within `train/agent/agent.py` is where we have the DQN Neural Network class and the DQAgent which is where the Neural Network is implemented to have replay and memory
    - Within `train/agent/schedule.py` is where the hyperparameter scheduling schemes were specified
- The notebooks folder which contains all the notebooks where all the tests, tuning, and prototyping were done for this project

### Executing program
* Edit `train/cfgs/config.yml` to set hyperparameters
* Run `train/train.py` to start training



## Model Information

### Architecture


    


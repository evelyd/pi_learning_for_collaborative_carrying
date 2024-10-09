# Physics-informed trajectory generator for collaborative payload carrying for humanoid robots

## Usage

### Install

To run the code in this repo, you can clone the repo:
```
git clone https://github.com/evelyd/pi_learning_for_collaborative_carrying.git
cd pi_learning_for_collaborative_carrying/
```
and create a conda environment to install the package:
```
conda env create -n pi_collab --file environment.yml
conda activate pi_collab
pip install .
```

### Run the code
Then within the conda env you will be able to run the scripts in the `scripts` folder, e.g.
```
cd scripts/
python retarget_onto_robot
```

## Responsible:
|     [Evelyn D'Elia](https://github.com/evelyd)    |
|-------------------------------------------------------|
|<img src="https://github.com/evelyd.png" width="180">|

## Background
Human-robot collaboration is an important field of research for humanoid robots. One particularly useful application of a humanoid robot is to alleviate physical strain on human workers, for instance by helping the human to carry heavy or ungainly payloads. Historically, there has been a significant amount of research into this type of task, but very little work has been done on leveraging machine learning methods. In particular, physics-informed ML has significant potential for 'baking' information about the physical system into the learning process in order to guide it. This can be extremely useful to reduce the amount of training data and to reduce the time it takes to converge to an acceptable result.

## Goal
This project aims to equip the ergoCub humanoid robot with the ability to collaboratively carry a payload with a human by taking cues from the human's movements. The human will wear the iFeel sensor suit, allowing the robot to be aware of the human's state at any given time. A learning-based method will take the human state as input and output a trajectory for the robot to follow such that it works with the human.

## Milestones
* Design a transformer-based framework with a physics-informed loss which allows the robot to follow the human while facing the human
* Collect collaborative lifting data
* Train the model using the human state as input and the predicted robot trajectory as output
* Validate the model performance in simulation and on the real robot


# Learning Task-Agnostic Action Spaces for Movement Optimization

This repository contains the source code for the algorithm, described in [this paper](https://arxiv.org/abs/2009.10337).

## Abstract
We propose a novel method for exploring the dynamics of physically based animated characters, and learning a task-agnostic action space that makes movement optimization easier. Like several previous papers, we parameterize actions as target states, and learn a short-horizon goal-conditioned low-level control policy that drives the agent's state towards the targets. Our novel contribution is that with our exploration data, we are able to learn the low-level policy in a generic manner and without any reference movement data. Trained once for each agent or simulation environment, the policy improves the efficiency of optimizing both trajectories and high-level policies across multiple tasks and optimization algorithms. We also contribute novel visualizations that show how using target states as actions makes optimized trajectories more robust to disturbances; this manifests as wider optima that are easy to find. Due to its simplicity and generality, our proposed approach should provide a building block that can improve a large variety of movement optimization methods and applications.

## Installation

## Prerequisites
- Python 3.5 or above
- cma
- glfw
- gym
- Keras
- mujoco-py
- numpy
- opencv-python
- pandas
- Pillow
- stable-baselines
- tensorflow
More detailed requirements are specified in ```requirements.txt```.

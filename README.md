# Online Adaptation for Enhancing Imitation Learning Policies

This repo contains the minimal requirements and files to reproduce the experiments conducted in the related [paper](link).

## How to install
Simply clone this repo and run `pip install -r requirements.txt`

# How to use
First, collect some trajectories using `play.py`. The script requires you to specify a save path for the recorded trajectories.
To train an agent, use the related agent file. For example, to train a GAIL agent, use `train_gail.py`.
To test an agent, run the related agent file. As an example, if you want to test BC, use `test_bc.py`.

# Known issues and TODOs
- TODO: Re-organize files

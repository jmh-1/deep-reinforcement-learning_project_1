# Project 1: Navigation

## Project Details

This project trains various DQN agents to navigate and collect bananas in a large, square world. The state space has 37 continuous values and there are 4 discrete actions:
> - **`0`** - move forward.
> - **`1`** - move backward.
> - **`2`** - turn left.
> - **`3`** - turn right.

The task is episodic and is considered solved when an agent gets an average score of at least 13 over 100 consecutive episodes
## Getting Started

Running the project requires the following python dependencies which can be installed with pip 

		Pillow>=4.2.1
		matplotlib
		numpy>=1.11.0
		jupyter
		pytest>=3.2.2
		docopt
		pyyaml
		protobuf==3.5.2
		grpcio
		torch
		pandas
		scipy
		ipykernel

It also requires the banana environment, which can be dowloaded [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip), unzip this file in the p1_navigation directory. 

Alternately, it can be run in the provided Udacity workspace, after installing dependencies with this command:

		pip -q install ./python

Test_Agents.ipynb can be used to test each of the agents created for this project, by running the cells in order. To run in the Udacity workspace, the file name provided to the UnityEnvironment constructor must be changed to 

		/data/Banana_Linux_NoVis/Banana.x86_64

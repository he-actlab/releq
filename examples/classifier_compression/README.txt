*** under development ***

Project Title: 
---------------------------------
ReLeQ: An Automatic Reinforcement Learning Approach
for Deep Quantization of Neural Networks
https://arxiv.org/abs/1811.01704

Usage:
---------------------------------
1. Update the config yaml file: releq_config.yaml
	a) Number of episodes: [int]
	b) Supported bitwidths: [list]
2. Run releq_<network_name>.py
	$ python3 releq_<network_name>.py

* supported networks: {lenet, cifar, svhn}

Running reward per network:
-----------------------------------
1. Based on which network you want to run, copy the network info from layer_info.py and paste it at the end of ppo_all_layers.py
2. After that just run "python3 ppo_all_layers.py"

Requirements:
---------------------------------
- Python3
- Tensorflow
- Pytorch 
- Distiller 
https://github.com/NervanaSystems/distiller

License:
---------------------------------








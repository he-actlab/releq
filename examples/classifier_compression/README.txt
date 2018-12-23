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

Requirements:
---------------------------------
- Python3
- Tensorflow
- Pytorch 
- Distiller 
https://github.com/NervanaSystems/distiller

License:
---------------------------------
MIT License









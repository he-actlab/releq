#---------------------------------------------------------------------------------------
network_name = "svhn"
number_of_layers = 8
file_name = "releq_svhn_learning_history_log.csv"
layer_info = StringIO("""layer_idx_norm;n;c;k;std
1;32;3;3;0.18325
2;32;32;3;0.04787
3;64;32;3;0.04403
4;64;64;3;0.03448
5;128;64;3;0.03441
6;128;128;3;0.02876
7;256;128;3;0.02559
8;10;256;0;0.09887""")
with open(file_name, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(headers)
layer_state_info = pandas.read_csv(layer_info, sep=";")
min_n = min(layer_state_info.loc[:, 'n'])
max_n = max(layer_state_info.loc[:, 'n'])
min_c = min(layer_state_info.loc[:, 'c'])
max_c = max(layer_state_info.loc[:, 'c'])
min_k = min(layer_state_info.loc[:, 'k'])
max_k = max(layer_state_info.loc[:, 'k'])
for layer in range(number_of_layers):
    layer_state_info.loc[layer, 'n'] = (layer_state_info.loc[layer, 'n'] - min_n)/(max_n - min_n)
    layer_state_info.loc[layer, 'c'] = (layer_state_info.loc[layer, 'c'] - min_c)/(max_c - min_c)
    layer_state_info.loc[layer, 'k'] = (layer_state_info.loc[layer, 'k'] - min_k)/(max_k - min_k)
print(layer_state_info)
layer_names = ["features.0", "features.3", "features.7", "features.10", "features.14", "features.17", "features.21", "classifier.0"]
training_cmd = "python3 compress_classifier.py --arch svhn ../../../data.svhn --quantize-eval --compress svhn_bn_wrpn.yaml --epochs 5 --lr 0.01 --resume svhn.pth.tar"
yaml_file = "svhn_bn_wrpn.yaml"
accuracy_cache_file = "svhn_accuracy_cache.txt"
quant_type = "wrpn_quantizer"
rl_quant = RLQuantization(number_of_layers, 97, network_name, layer_names, layer_state_info, training_cmd, yaml_file, quant_type) #num_layers, accuracy, network_name, layer_names, layer_stats
rl_quant.quantize_layers_together(number_of_layers)
#-----------------------------------------------------------------------------------------
network_name = "resnet-20"
number_of_layers = 20
file_name = "releq_resnet20_learning_history_log.csv"
layer_info = StringIO("""layer_idx_norm;n;c;k;std
2;16;16;3;0.13231
3;16;16;3;0.12662
4;16;16;3;0.12674
5;16;16;3;0.12042
6;16;16;3;0.12214
7;16;16;3;0.11869
8;32;16;3;0.09128
9;32;32;3;0.08443
10;32;16;1;0.23298
11;32;32;3;0.08471
12;32;32;3;0.08416
13;32;32;3;0.08387
14;32;32;3;0.08293
15;64;32;3;0.06192
16;64;64;3;0.06001
17;64;32;1;0.16850
18;64;64;3;0.06001
19;64;64;3;0.05707
20;64;64;3;0.05611
21;64;64;3;0.05470
""")
with open(file_name, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(headers)
layer_state_info = pandas.read_csv(layer_info, sep=";")
min_n = min(layer_state_info.loc[:, 'n'])
max_n = max(layer_state_info.loc[:, 'n'])
min_c = min(layer_state_info.loc[:, 'c'])
max_c = max(layer_state_info.loc[:, 'c'])
min_k = min(layer_state_info.loc[:, 'k'])
max_k = max(layer_state_info.loc[:, 'k'])
for layer in range(number_of_layers):
    layer_state_info.loc[layer, 'n'] = (layer_state_info.loc[layer, 'n'] - min_n)/(max_n - min_n)
    layer_state_info.loc[layer, 'c'] = (layer_state_info.loc[layer, 'c'] - min_c)/(max_c - min_c)
    layer_state_info.loc[layer, 'k'] = (layer_state_info.loc[layer, 'k'] - min_k)/(max_k - min_k)
print(layer_state_info)
layer_names = ["layer1.0.conv1", "layer1.0.conv2", "layer1.1.conv1", "layer1.1.conv2", "layer1.2.conv1", "layer1.2.conv2", "layer2.0.conv1", "layer2.0.conv2", "layer2.0.downsample.0", "layer2.1.conv1", "layer2.1.conv2", "layer2.2.conv1", "layer2.2.conv2", "layer3.0.conv1", "layer3.0.conv2", "layer3.0.downsample.0", "layer3.1.conv1", "layer3.1.conv2", "layer3.2.conv1", "layer3.2.conv2"]
training_cmd = "python3 compress_classifier.py --arch resnet20_cifar ../../../data.cifar --epochs 10 --lr 0.001 --resume resnet.pth.tar --compress resnet_bn_dorefa.yaml"
yaml_file = "resnet_bn_dorefa.yaml"
accuracy_cache_file = "resnet20_accuracy_cache.txt"
quant_type = "dorefa_quantizer"
rl_quant = RLQuantization(number_of_layers, 93, network_name, layer_names, layer_state_info, training_cmd, yaml_file, quant_type) #num_layers, accuracy, network_name, layer_names, layer_stats
rl_quant.quantize_layers_together(number_of_layers)
#-----------------------------------------------------------------------------------------
network_name = "lenet"
number_of_layers = 4
file_name = "releq_lenet_learning_history_log.csv"
layer_info = StringIO("""layer_idx_norm;n;c;k;std
1;20;1;5;0.18183
2;50;20;5;0.03791
3;500;800;0;0.02124
4;10;500;0;0.06587""")
with open(file_name, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(headers)
layer_state_info = pandas.read_csv(layer_info, sep=";")
min_n = min(layer_state_info.loc[:, 'n'])
max_n = max(layer_state_info.loc[:, 'n'])
min_c = min(layer_state_info.loc[:, 'c'])
max_c = max(layer_state_info.loc[:, 'c'])
min_k = min(layer_state_info.loc[:, 'k'])
max_k = max(layer_state_info.loc[:, 'k'])
for layer in range(number_of_layers):
    layer_state_info.loc[layer, 'n'] = (layer_state_info.loc[layer, 'n'] - min_n)/(max_n - min_n)
    layer_state_info.loc[layer, 'c'] = (layer_state_info.loc[layer, 'c'] - min_c)/(max_c - min_c)
    layer_state_info.loc[layer, 'k'] = (layer_state_info.loc[layer, 'k'] - min_k)/(max_k - min_k)
print(layer_state_info)
layer_names = ["conv1", "conv2", "fc1", "fc2"]
training_cmd = "python3 compress_classifier.py --arch lenet_mnist ../../../data.mnist --quantize-eval --compress ./lenet_bn_dorefa.yaml --epochs 5 --lr 0.001 --resume ./lenet_mnist.pth.tar"
yaml_file = "lenet_bn_dorefa.yaml"
accuracy_cache_file = "lenet_accuracy_cache.txt"
quant_type = "dorefa_quantizer"
rl_quant = RLQuantization(number_of_layers, 99.8, network_name, layer_names, layer_state_info, training_cmd, yaml_file, quant_type) #num_layers, accuracy, network_name, layer_names, layer_stats
rl_quant.quantize_layers_together(number_of_layers)
#-------------------------------------------------------------------------------------------
network_name = "cifar10"
file_name = "releq_cifar10_learning_history_log.csv"
number_of_layers = 5 #cifar
layer_info = StringIO("""layer_idx_norm;n;c;k;std;c_out
1;6;3;5;0.23009;6
2;16;6;5;0.11020;400
3;120;400;0;0.04013;120
4;84;120;0;0.06537;84
5;10;84;0;0.14734;1""")
with open(file_name, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(headers)
layer_state_info = pandas.read_csv(layer_info, sep=";")
min_n = min(layer_state_info.loc[:, 'n'])
max_n = max(layer_state_info.loc[:, 'n'])
min_c = min(layer_state_info.loc[:, 'c'])
max_c = max(layer_state_info.loc[:, 'c'])
min_k = min(layer_state_info.loc[:, 'k'])
max_k = max(layer_state_info.loc[:, 'k'])
for layer in range(number_of_layers):
    layer_state_info.loc[layer, 'n'] = (layer_state_info.loc[layer, 'n'] - min_n)/(max_n - min_n)
    layer_state_info.loc[layer, 'c'] = (layer_state_info.loc[layer, 'c'] - min_c)/(max_c - min_c)
    layer_state_info.loc[layer, 'k'] = (layer_state_info.loc[layer, 'k'] - min_k)/(max_k - min_k)
print(layer_state_info)
layer_names = ["conv1", "conv2", "fc1", "fc2", "fc3"]
training_cmd = "python3 compress_classifier.py --arch simplenet_cifar ../../../data.cifar --quantize-eval --compress cifar_bn_wrpn.yaml --epochs 10 --lr 0.001 --resume ./simplenet_cifar.pth.tar"
yaml_file = "cifar_bn_wrpn.yaml"
accuracy_cache_file = "cifar_accuracy_cache.txt"
quant_type = "wrpn_quantizer"
rl_quant = RLQuantization(number_of_layers, 75, network_name, layer_names, layer_state_info, training_cmd, yaml_file, quant_type) #num_layers, accuracy, network_name, layer_names, layer_stats
rl_quant.quantize_layers_together(number_of_layers)
#-------------------------------------------------------------------------------------------
network_name = "alexnet"
number_of_layers = 6
file_name = "releq_alexnet_learning_history_log.csv"
layer_info = StringIO("""layer_idx_norm;n;c;k;std
2;192;64;5;0.04702
3;384;192;3;0.03492
4;256;384;3;0.2750
5;256;256;3;0.02619
6;4096;9216;0;0.00928
7;4096;4096;0;0.01157
""")
with open(file_name, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(headers)
layer_state_info = pandas.read_csv(layer_info, sep=";")
min_n = min(layer_state_info.loc[:, 'n'])
max_n = max(layer_state_info.loc[:, 'n'])
min_c = min(layer_state_info.loc[:, 'c'])
max_c = max(layer_state_info.loc[:, 'c'])
min_k = min(layer_state_info.loc[:, 'k'])
max_k = max(layer_state_info.loc[:, 'k'])
for layer in range(number_of_layers):
    layer_state_info.loc[layer, 'n'] = (layer_state_info.loc[layer, 'n'] - min_n)/(max_n - min_n)
    layer_state_info.loc[layer, 'c'] = (layer_state_info.loc[layer, 'c'] - min_c)/(max_c - min_c)
    layer_state_info.loc[layer, 'k'] = (layer_state_info.loc[layer, 'k'] - min_k)/(max_k - min_k)
print(layer_state_info)
layer_names = ["features.3", "features.6", "features.8", "features.10", "classifier.1", "classifier.4"]
training_cmd = "python3 compress_classifier.py --arch alexnet ../../../data.imagenet_100 --epochs 2 --resume alexnet.pth.tar --lr 0.001 --quantize-eval --compress alexnet_bn_dorefa.yaml"
yaml_file = "alexnet_bn_dorefa.yaml"
accuracy_cache_file = "alexnet_accuracy_cache.txt"
quant_type = "dorefa_quantizer"
rl_quant = RLQuantization(number_of_layers, 82.78, network_name, layer_names, layer_state_info, training_cmd, yaml_file, quant_type) #num_layers, accuracy, network_name, layer_names, layer_stats
rl_quant.quantize_layers_together(number_of_layers)
#-------------------------------------------------------------------------------------------
network_name = "vgg11"
number_of_layers = 7
file_name = "releq_vgg11_learning_history_log.csv"
layer_info = StringIO("""layer_idx_norm;n;c;k;std
0;128;64;3;0.04522
1;256;128;3;0.03049
2;256;256;3;0.02952
3;512;256;3;0.02094
4;512;512;3;0.02054
5;512;512;3;0.02046
6;512;512;3;0.02040
""")
with open(file_name, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(headers)
layer_state_info = pandas.read_csv(layer_info, sep=";")
min_n = min(layer_state_info.loc[:, 'n'])
max_n = max(layer_state_info.loc[:, 'n'])
min_c = min(layer_state_info.loc[:, 'c'])
max_c = max(layer_state_info.loc[:, 'c'])
min_k = min(layer_state_info.loc[:, 'k'])
max_k = max(layer_state_info.loc[:, 'k'])
for layer in range(number_of_layers):
    layer_state_info.loc[layer, 'n'] = (layer_state_info.loc[layer, 'n'] - min_n)/(max_n - min_n)
    layer_state_info.loc[layer, 'c'] = (layer_state_info.loc[layer, 'c'] - min_c)/(max_c - min_c)
    layer_state_info.loc[layer, 'k'] = (layer_state_info.loc[layer, 'k'])/2.0
print(layer_state_info)
layer_names = ["features.3", "features.6", "features.8", "features.11", "features.13", "features.16", "features.18"]
training_cmd = "python3 compress_classifier.py --arch vgg11_cifar ../../../data.cifar --epochs 10 --lr 0.0001 --resume vgg11.pth.tar --compress vgg11_bn_wrpn.yaml"
yaml_file = "vgg11_bn_wrpn.yaml"
accuracy_cache_file = "vgg11_accuracy_cache.txt"
quant_type = "wrpn_quantizer"
rl_quant = RLQuantization(number_of_layers, 96, network_name, layer_names, layer_state_info, training_cmd, yaml_file, quant_type) #num_layers, accuracy, network_name, layer_names, layer_stats
rl_quant.quantize_layers_together(number_of_layers)
#-------------------------------------------------------------------------------------------
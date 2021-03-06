#!/usr/bin/python

#Prannoyy Pilligundla
# (ppilligu@eng.ucsd.edu)

import sys
import copy
from collections import OrderedDict
import os
import shutil
import subprocess
import time
from subprocess import Popen, PIPE, STDOUT
import shlex
from copy import deepcopy
import pandas
import random
import yaml
from io import StringIO
##
# TensorFlow Policy Gradient
##
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import csv

try:
    xrange = xrange
except:
    xrange = range
#############################

import logging
# AHMED: LOGGING
LOG_FILENAME = 'releq.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG, filemode='w')

#############################

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
pass

GAMMA = 0.90

class Policy_net:
    def __init__(self, name: str, temp=0.5, num_actions=3):
        """
        :param name: string
        :param env: gym env
        :param temp: temperature of boltzmann distribution
        """

        #ob_space = env.observation_space
        #act_space = env.action_space

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None] + list([7]), name='obs')

            layer_1 = tf.layers.dense(inputs=self.obs, units=128, activation=tf.nn.relu, name = 'layer_1')
            with tf.variable_scope('layer_1', reuse=True):
                    self.layer_1_weights = tf.get_variable('kernel')

            with tf.variable_scope('policy_net'):
                #layer_1 = tf.layers.dense(inputs=self.obs, units=200, activation=tf.nn.relu)
                layer_2 = tf.layers.dense(inputs=layer_1, units=128, activation=tf.nn.relu, name = 'policy_layer_2')
                layer_3 = tf.layers.dense(inputs=layer_2, units=128, activation=tf.nn.relu, name = 'policy_layer_3')
                self.act_probs = tf.layers.dense(inputs=tf.divide(layer_3, temp), units=num_actions, activation=tf.nn.softmax)
                with tf.variable_scope('policy_layer_2', reuse=True):
                    self.layer_2_weights = tf.get_variable('kernel')
                with tf.variable_scope('policy_layer_3', reuse=True):
                    self.layer_3_weights = tf.get_variable('kernel')

            with tf.variable_scope('value_net'):
                #layer_1 = tf.layers.dense(inputs=self.obs, units=40, activation=tf.nn.relu)
                layer_2 = tf.layers.dense(inputs=layer_1, units=128, activation=tf.nn.relu)
                layer_3 = tf.layers.dense(inputs=layer_2, units=64, activation=tf.nn.relu)
                self.v_preds = tf.layers.dense(inputs=layer_3, units=1, activation=None)

            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            self.scope = tf.get_variable_scope().name

    def act(self, obs, stochastic=True):
        if stochastic:
            return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs})
        else:
            return tf.get_default_session().run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs})

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs})
    
    def get_policy_weights(self):
        return tf.get_default_session().run([self.layer_1_weights, self.layer_2_weights, self.layer_3_weights])

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class RPolicy_net:
    def __init__(self, name: str, lstm_cell, temp=0.5, num_actions=3):
        """
        :param name: string
        :param env: gym env
        :param temp: temperature of boltzmann distribution
        """

        #ob_space = env.observation_space
        #act_space = env.action_space
        self.rnn_initial_state_in = lstm_cell.zero_state(1, tf.float32) #Argument 1 is the batch_size
        self.rnn_state_in = lstm_cell.zero_state(1, tf.float32)

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None] + list([8]), name='obs')

            obs_rnn_input = tf.reshape(self.obs, [1, 1, 8])
            self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=obs_rnn_input, cell=lstm_cell, dtype=tf.float32, initial_state=self.rnn_state_in, scope='Policy_rnn')
            self.rnn = tf.reshape(self.rnn,shape=[-1, 128])

            #layer_1 = tf.layers.dense(inputs=self.obs, units=128, activation=tf.nn.relu, name = 'layer_1')

            #with tf.variable_scope('layer_1', reuse=True):
            #        self.layer_1_weights = tf.get_variable('kernel')

            with tf.variable_scope('policy_net'):
                #layer_1 = tf.layers.dense(inputs=self.obs, units=200, activation=tf.nn.relu)
                layer_2 = tf.layers.dense(inputs=self.rnn, units=128, activation=tf.nn.relu, name = 'policy_layer_2')
                layer_3 = tf.layers.dense(inputs=layer_2, units=128, activation=tf.nn.relu, name = 'policy_layer_3')
                self.act_probs = tf.layers.dense(inputs=tf.divide(layer_3, temp), units=num_actions, activation=tf.nn.softmax)
                with tf.variable_scope('policy_layer_2', reuse=True):
                    self.layer_2_weights = tf.get_variable('kernel')
                with tf.variable_scope('policy_layer_3', reuse=True):
                    self.layer_3_weights = tf.get_variable('kernel')

            with tf.variable_scope('value_net'):
                #layer_1 = tf.layers.dense(inputs=self.obs, units=40, activation=tf.nn.relu)
                layer_2 = tf.layers.dense(inputs=self.rnn, units=128, activation=tf.nn.relu)
                layer_3 = tf.layers.dense(inputs=layer_2, units=64, activation=tf.nn.relu)
                self.v_preds = tf.layers.dense(inputs=layer_3, units=1, activation=None)

            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            self.scope = tf.get_variable_scope().name

    def act(self, obs, stochastic=True):
        if stochastic:
            return tf.get_default_session().run([self.act_stochastic, self.v_preds, self.rnn_state], feed_dict={self.obs: obs})
        else:
            return tf.get_default_session().run([self.act_deterministic, self.v_preds, self.rnn_state], feed_dict={self.obs: obs})

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs})
    
    def get_policy_weights(self):
        return tf.get_default_session().run([self.layer_2_weights, self.layer_3_weights])

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class PPOTrain:
    def __init__(self, Policy, Old_Policy, gamma=0.95, clip_value=0.3, c_1=1, c_2=0.01):
        """
        :param Policy:
        :param Old_Policy:
        :param gamma:
        :param clip_value:
        :param c_1: parameter for value difference
        :param c_2: parameter for entropy bonus
        """

        self.Policy = Policy
        self.Old_Policy = Old_Policy
        self.gamma = gamma

        pi_trainable = self.Policy.get_trainable_variables()
        old_pi_trainable = self.Old_Policy.get_trainable_variables()

        # assign_operations for policy parameter values to old policy parameters
        with tf.variable_scope('assign_op'):
            self.assign_ops = []
            for v_old, v in zip(old_pi_trainable, pi_trainable):
                self.assign_ops.append(tf.assign(v_old, v))

        # inputs for train_op
        with tf.variable_scope('train_inp'):
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
            self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')

        act_probs = self.Policy.act_probs
        act_probs_old = self.Old_Policy.act_probs

        # probabilities of actions which agent took with policy
        act_probs = act_probs * tf.one_hot(indices=self.actions, depth=act_probs.shape[1])
        act_probs = tf.reduce_sum(act_probs, axis=1)

        # probabilities of actions which agent took with old policy
        act_probs_old = act_probs_old * tf.one_hot(indices=self.actions, depth=act_probs_old.shape[1])
        act_probs_old = tf.reduce_sum(act_probs_old, axis=1)

        with tf.variable_scope('loss/clip'):
            # ratios = tf.divide(act_probs, act_probs_old)
            ratios = tf.exp(tf.log(act_probs) - tf.log(act_probs_old))
            tf.summary.scalar('ratios', ratios[0])
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - clip_value, clip_value_max=1 + clip_value)
            loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
            loss_clip = tf.reduce_mean(loss_clip)
            tf.summary.scalar('loss_clip', loss_clip)

        # construct computation graph for loss of value function
        with tf.variable_scope('loss/vf'):
            v_preds = self.Policy.v_preds
            loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_preds_next, v_preds)
            loss_vf = tf.reduce_mean(loss_vf)
            tf.summary.scalar('loss_vf', loss_vf)

        # construct computation graph for loss of entropy bonus
        with tf.variable_scope('loss/entropy'):
            entropy = -tf.reduce_sum(self.Policy.act_probs *
                                     tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0)), axis=1)
            entropy = tf.reduce_mean(entropy, axis=0)  # mean of entropy of pi(obs)
            tf.summary.scalar('entropy', entropy)

        with tf.variable_scope('loss'):
            loss = loss_clip - c_1 * loss_vf + c_2 * entropy
            loss = -loss  # minimize -loss == maximize loss
            tf.summary.scalar('loss', loss)

        self.merged = tf.summary.merge_all()
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)
        self.train_op = optimizer.minimize(loss, var_list=pi_trainable, global_step=tf.train.get_global_step())

    def train(self, obs, actions, rewards, v_preds_next, gaes):
        tf.get_default_session().run([self.train_op], feed_dict={self.Policy.obs: obs,
                                                                 self.Old_Policy.obs: obs,
                                                                 self.actions: actions,
                                                                 self.rewards: rewards,
                                                                 self.v_preds_next: v_preds_next,
                                                                 self.gaes: gaes})

    def get_summary(self, obs, actions, rewards, v_preds_next, gaes):
        return tf.get_default_session().run([self.merged], feed_dict={self.Policy.obs: obs,
                                                                      self.Old_Policy.obs: obs,
                                                                      self.actions: actions,
                                                                      self.rewards: rewards,
                                                                      self.v_preds_next: v_preds_next,
                                                                      self.gaes: gaes})

    def assign_policy_parameters(self):
        # assign policy parameter values to old policy parameters
        return tf.get_default_session().run(self.assign_ops)

    def get_gaes(self, rewards, v_preds, v_preds_next):
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.gamma * gaes[t + 1]
        return gaes


class RLQuantization:
    def __init__(self, num_layers, accuracy, network_name, layer_names, layer_state_info):
        
        self.yaml_config_file = "releq_config.yaml"
        with open(self.yaml_config_file) as f:
            self.yaml_config = yaml.load(f)
        
        self.num_layers = num_layers # number of layers in the NN that needs to be Optimized
        self.n_act_p_episode     = 1  # number of actions per each episod (fix for now)
        #self.total_episodes        = num_episodes  # total number of observations used for training (in order)
        self.total_episodes = 700
        self.network_name     = network_name  # defines the network name

        #self.supported_bit_widths = self.yaml_config["supported_bitwidths"] #[2, 3, 4, 5, 8] #[2, 3, 4, 5, 8]
        self.supported_bit_widths = [2, 3, 4, 5, 8]
        self.max_bitwidth = max(self.supported_bit_widths)
        self.min_bitwidth = min(self.supported_bit_widths)

        """ Clear the TensorFlow graph """
        tf.reset_default_graph() 

        policy_lstm_cell = tf.contrib.rnn.LSTMCell(num_units=128, state_is_tuple=True) #tf.contrib.cudnn_rnn.CudnnLSTM(1, num_units=4, kernel_initializer=tf.contrib.layers.xavier_initializer())
        old_policy_lstm_cell = tf.contrib.rnn.LSTMCell(num_units=128, state_is_tuple=True) #tf.contrib.cudnn_rnn.CudnnLSTM(1, num_units=4, kernel_initializer=tf.contrib.layers.xavier_initializer())

        """ Load the agent """
        #self.Policy = RPolicy_net('policy', policy_lstm_cell, num_actions=len(self.supported_bit_widths))
        #self.Old_Policy = RPolicy_net('old_policy', old_policy_lstm_cell, num_actions=len(self.supported_bit_widths))
        self.Policy = Policy_net('policy', num_actions=len(self.supported_bit_widths))
        self.Old_Policy = Policy_net('old_policy', num_actions=len(self.supported_bit_widths))
        self.PPO = PPOTrain(self.Policy, self.Old_Policy, gamma=GAMMA)
        self.saver = tf.train.Saver()

        self.quant_state = 1
        self.acc_reward_const = 1
        self.quant_reward_const = 0.2

        self.fp_accuracy = accuracy
        self.layer_state_info = layer_state_info
        self.layer_names = layer_names

        #self.yaml_file = "cifar_bn_wrpn.yaml.yaml"
        self.yaml_file = "mobilenet_bn_wrpn.yaml"
        with open(self.yaml_file) as f:
            self.yaml_out = yaml.load(f)
        
    
    def quantize_layers(self):
        """ delete the training for current network """
        train_dir = "/backup/amir-tc/rlquant.code/rl-training"
        network_dirname = train_dir + "/" + self.network_name
        if os.path.exists(network_dirname):
            shutil.rmtree(network_dirname)
        os.makedirs(network_dirname)

        """ Start TensorFlow """
        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)

        """ Launch the TensorFlow graph """
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(init)

            cur_accuracy = self.fp_accuracy
            bitwidth_layers = [8 for i in range(self.num_layers)]

            for i in range(self.total_episodes):
                print(bcolors.OKGREEN + "# Running epidode %d..." % (i) + bcolors.ENDC)

                if (self.total_episodes - i) < 10:
                    stoch = False
                    print("taking deterministic actions") 
                else:
                    stoch = True 

                for layer in range(self.num_layers): #Not Quantizing the first and last layers
                    new_bitwidth, cur_accuracy = self.quantize_layer(i, layer, bitwidth_layers, self.quant_state, cur_accuracy)
                    bitwidth_layers[layer] = new_bitwidth
                    self.update_quant_state(bitwidth_layers)
                    # here: quant_state
                    print("EPISODE-", i, "LAYER-", layer, " ACC:", cur_accuracy, " NEW_BITWIDTH:", new_bitwidth)
            
                print("End of Episode ", i,", quantized bitwidths ", bitwidth_layers, " Quant_State ", self.quant_state)
                print("Accuracy with new bit_widths is ", cur_accuracy)
            return bitwidth_layers, cur_accuracy  
            
    
    def quantize_layer(self, episode_num, layer_num, bitwidth_layers, quant_state, accuracy):
        #Building State
        global acc_cache
        intial_layer_state = [self.layer_state_info.loc[layer_num, 'layer_idx_norm'], bitwidth_layers[layer_num]/32, quant_state, accuracy/self.fp_accuracy, self.layer_state_info.loc[layer_num, 'n'], self.layer_state_info.loc[layer_num, 'c'], self.layer_state_info.loc[layer_num, 'k'], self.layer_state_info.loc[layer_num, 'std']]
        cur_accuracy = accuracy
        prev_accuracy = accuracy
        new_bitwidth = bitwidth_layers[layer_num]

        s = intial_layer_state
        s_history  = []
        a_history  = []
        rewards  = []
        v_preds = []

        writer = tf.summary.FileWriter('./log/train', tf.get_default_session().graph)

        # LSTM adjustment for resetting or not!
        #if layer_num == 1:
        #    self.Policy.rnn_state_in = self.Policy.rnn_initial_state_in

        for i in range(self.n_act_p_episode):
            print(bcolors.OKGREEN + "# Running action %d for layer %d..." % (i, layer_num) + bcolors.ENDC)
        
            #act_index: 0-> Dec Bits, 1-> Keep Same, 2->Inc Bits
            act_index, v_pred, self.Policy.rnn_state_in = self.Policy.act(obs=[s], stochastic=True)
            #act_index, v_pred = self.Policy.act(obs=[s], stochastic=True)
            print("Action Probabilities ", self.Policy.get_action_prob(obs=[s]))
            #l2w, l3w = self.Policy.get_policy_weights()
            #if np.isnan(l1w).any():
            #    print("L1W ", l1w)
            #    os._exit(0)
            #if np.isnan(l2w).any():
            #    print("L2W ", l2w)
            #    os._exit(0)
            #if np.isnan(l3w).any():
            #    print("L3W ", l3w)
            #    os._exit(0)
            #print("Gradients ", l1w, l2w, l3w)

            act_index = np.asscalar(act_index)
            v_pred = np.asscalar(v_pred)

            s_history.append(s)
            a_history.append(act_index)
            v_preds.append(v_pred)

            # AHMED: episode log
            #print("Action Probabilities ", self.Policy.get_action_prob(obs=[s]), s)
            data2 = self.Policy.get_action_prob(obs=[s])      

            cur_bitwidth = s[1]
            #new_bitwidth = self.perform_action(cur_bitwidth*32, act_index)
            new_bitwidth = self.perform_flexible_action(act_index)
            s[1] = new_bitwidth

            #Calculate Reward
            new_bitwidth_layers = deepcopy(bitwidth_layers)
            new_bitwidth_layers[layer_num] = new_bitwidth
            print("Bitwidth layers ", new_bitwidth_layers)
            self.update_yaml_file(new_bitwidth_layers)
            
            # acc_bw cache - CHECKING 
            if str(new_bitwidth_layers) in acc_cache:
               cur_accuracy = acc_cache[str(new_bitwidth_layers)]
            else:
               #os.system("python3 compress_classifier.py --arch svhn ../../../data.svhn --quantize-eval --compress ./svhn_bn_wrpn.yaml --epochs 1 --resume ./svhn.pth.tar --lr 0.001")
               os.system("python3 compress_classifier.py --arch svhn ../../../data.svhn --quantize-eval --compress ./svhn_bn_dorefa.yaml --epochs 1 --resume ./svhn.pth.tar --lr 0.001")
               cur_accuracy = float(open("val_accuracy.txt").readlines()[0])
               #cur_accuracy = self.nn_inference_func(self.network_name, new_bitwidth_layers) #self.nn_inference_func(self.network_name, episode_num, layer_num, new_bitwidth_layers)
               # acc-bw caching - CACHE UPDATE  
               acc_cache[str(new_bitwidth_layers)] = cur_accuracy  
                
            self.quant_reward_const = 1*cur_accuracy/self.fp_accuracy

            self.update_quant_state(new_bitwidth_layers)

            reward = self.calculate_reward_shaping(cur_accuracy)
            #reward = self.calculate_reward(cur_accuracy, self.fp_accuracy, bitwidth_layers[layer_num], new_bitwidth)

            s[3] = cur_accuracy/self.fp_accuracy # ACC state
            # AHMED: debug
            #data = [episode_num, layer_num, self.quant_state, s[3], reward]
            #data = [episode_num, layer_num, self.quant_state, s[3], reward, bitwidth_layers[0],bitwidth_layers[1],bitwidth_layers[2],bitwidth_layers[3],bitwidth_layers[4],bitwidth_layers[5],bitwidth_layers[6],bitwidth_layers[7]]
            data = [episode_num, layer_num, self.quant_state, s[3], reward]
            for each in new_bitwidth_layers:
                data.append(each)
            for each in data2[0]:
                data.append(each)
            write_to_csv(data)
            # ------------------------------------
            prev_accuracy = cur_accuracy
            rewards.append(reward)

        v_preds_next = v_preds[1:] + [0]

        gaes = self.PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)

        # convert list to numpy array for feeding tf.placeholder
        observations = np.reshape(s_history, newshape=[-1] + list([8]))
        actions = np.array(a_history).astype(dtype=np.int32)
        rewards = np.array(rewards).astype(dtype=np.float32)
        v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
        gaes = np.array(gaes).astype(dtype=np.float32)
        #gaes = (gaes - gaes.mean()) / gaes.std()
        
        self.PPO.assign_policy_parameters()
        
        inp = [observations, actions, rewards, v_preds_next, gaes]
        
        # train
        for epoch in range(1):
            sample_indices = np.random.randint(low=0, high=observations.shape[0], size=1)  # indices are in [low, high)
            sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
            print(sampled_inp)
            self.PPO.train(obs=sampled_inp[0], actions=sampled_inp[1], rewards=sampled_inp[2], v_preds_next=sampled_inp[3], gaes=sampled_inp[4])
        
        summary = self.PPO.get_summary(obs=inp[0], actions=inp[1], rewards=inp[2], v_preds_next=inp[3], gaes=inp[4])[0]
        
        writer.add_summary(summary, self.num_layers*episode_num+layer_num)
        writer.close()

        #Optional: Return Bit Width from the newly trained Policy
        #act_index, v_pred = self.Policy.act(obs=[s], stochastic=True)
        #new_bitwidth = self.perform_action(s[1], act_index)
        #new_bitwidth_layers = deepcopy(bitwidth_layers)
        #new_bitwidth_layers[layer_num] = new_bitwidth
        #print("New Bitwidth for layer ", layer_num, " is ", new_bitwidth)
        #cur_accuracy =  self.nn_inference_func(new_bitwidth_layers)

        return new_bitwidth, cur_accuracy        
    
    def perform_action(self, cur_bitwidth, act_index):
        #act_index: 0-> Dec Bits, 1-> Keep Same, 2->Inc Bits
        #supported_bit_widths = [1, 2, 4, 6, 8, 16, 32]
        supported_bit_widths = [4, 8, 16, 32]
        cur_bitwidth_index = supported_bit_widths.index(cur_bitwidth)
        if act_index == 0 and cur_bitwidth_index != 0:
            cur_bitwidth_index -= 1
        elif act_index == 2 and cur_bitwidth_index != (len(supported_bit_widths)-1):
            cur_bitwidth_index += 1
        return supported_bit_widths[cur_bitwidth_index]
    
    def perform_flexible_action(self, act_index):
        #act_index: Index of Bit Width Array
        return self.supported_bit_widths[act_index]
        
    def update_quant_state(self, bitwidth_layers):
        self.quant_state = sum(bitwidth_layers)/(max(self.supported_bit_widths)*self.num_layers)
    
    def update_yaml_file(self, weight_bitwidth_layers):
        
        #for i, element in enumerate(self.yaml_out["quantizers"]["wrpn_quantizer"]["bits_overrides"]):
        #    self.yaml_out["quantizers"]["wrpn_quantizer"]["bits_overrides"][element]["wts"] = weight_bitwidth_layers[i]

        for i, layer_name in enumerate(self.layer_names):
            self.yaml_out["quantizers"]["wrpn_quantizer"]["bits_overrides"][layer_name]["wts"] = weight_bitwidth_layers[i]
            #self.yaml_out["quantizers"]["dorefa_quantizer"]["bits_overrides"][layer_name]["wts"] = weight_bitwidth_layers[i]


        with open(self.yaml_file, "w") as f:
            yaml.dump(self.yaml_out, f, default_flow_style=False)
   

    def calculate_reward_shaping(self, cur_accuracy):
        margin = 0.7
        a = 0.8
        b = 1
        x_min = self.min_bitwidth/self.max_bitwidth
        x = self.quant_state - x_min  # QUANT state 
        acc_state = cur_accuracy/self.fp_accuracy # ACC state 
        y = acc_state
        reward = 1 - x**a
        if (y < margin):
            reward = -1
        else:
            acc_discount = (max(y, margin))**(b/max(y, margin))
            reward = 2*(reward * acc_discount - 0.5)
        return reward 
    
 
    def calculate_reward(self, cur_accuracy, prev_accuracy, cur_bitwidth, new_bitwidth):
        print("Acc Diff", cur_accuracy - prev_accuracy)
        acc_reward = (cur_accuracy - prev_accuracy)*self.acc_reward_const
        if cur_accuracy < self.fp_accuracy*0.5:
            if new_bitwidth > cur_bitwidth:
                quant_reward = -acc_reward + new_bitwidth - cur_bitwidth
            elif new_bitwidth == self.max_bitwidth and cur_bitwidth == self.max_bitwidth:
                quant_reward = -acc_reward
            else:
                quant_reward = acc_reward/2
        elif new_bitwidth == self.max_bitwidth and cur_bitwidth == self.max_bitwidth:
            quant_reward = -20*self.quant_state
        else:
            quant_reward = (cur_bitwidth - new_bitwidth)*self.quant_reward_const
        #prev_bitwidth = cur_bitwidth
        #cur_bitwidth = new_bitwidth
        #quant_reward = (cur_bitwidth - prev_bitwidth)*self.quant_reward_const
        total_reward = acc_reward+quant_reward
        total_reward /= 400
        print(bcolors.OKGREEN + "# total_reward %f , # quant_reward %f , acc_reward %f " % (total_reward, quant_reward, acc_reward) + bcolors.ENDC)
        return total_reward

    def quantize_layers_together(self, num_layers_together):
        """ delete the training for current network """
        train_dir = "/backup/amir-tc/rlquant.code/rl-training"
        network_dirname = train_dir + "/" + self.network_name
        if os.path.exists(network_dirname):
            shutil.rmtree(network_dirname)
        os.makedirs(network_dirname)

        """ Start TensorFlow """
        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)

        """ Launch the TensorFlow graph """
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(init)

            cur_accuracy = self.fp_accuracy

            for i in range(self.total_episodes):
                bitwidth_layers = [8 for i in range(self.num_layers)]
                print(bcolors.OKGREEN + "# Running epidode %d..." % (i) + bcolors.ENDC)
                s_history  = []
                a_history  = []
                rewards  = []
                v_preds = []

                for layer_num in range(self.num_layers):
                    intial_layer_state = [self.layer_state_info.loc[layer_num, 'layer_idx_norm'], bitwidth_layers[layer_num]/32, self.quant_state, self.layer_state_info.loc[layer_num, 'n'], self.layer_state_info.loc[layer_num, 'c'], self.layer_state_info.loc[layer_num, 'k'], self.layer_state_info.loc[layer_num, 'std']]

                    s = intial_layer_state

                    writer = tf.summary.FileWriter('./log/train', tf.get_default_session().graph)

                    print(bcolors.OKGREEN + "# Running action for layer %d..." % (layer_num) + bcolors.ENDC)

                    #if layer_num == 1:
                    #    self.Policy.rnn_state_in = self.Policy.rnn_initial_state_in
                
                    #act_index: 0-> Dec Bits, 1-> Keep Same, 2->Inc Bits
                    act_index, v_pred = self.Policy.act(obs=[s], stochastic=True)
                    print("Action Probabilities ", self.Policy.get_action_prob(obs=[s]))
                    l1w, l2w, l3w = self.Policy.get_policy_weights()
                    if np.isnan(l1w).any():
                        print("L1W ", l1w)
                        exit()
                    if np.isnan(l2w).any():
                        print("L2W ", l2w)
                        exit()
                    if np.isnan(l3w).any():
                        print("L3W ", l3w)
                        exit()
                    #print("Gradients ", l1w, l2w, l3w)
                    act_index = np.asscalar(act_index)
                    v_pred = np.asscalar(v_pred)

                    s_history.append(s)
                    a_history.append(act_index)
                    v_preds.append(v_pred)

                    new_bitwidth = self.perform_flexible_action(act_index)

                    #Calculate Reward
                    bitwidth_layers[layer_num] = new_bitwidth
                    print("Bitwidth layers ",  bitwidth_layers)
                    self.update_yaml_file(bitwidth_layers)
                    #s[3] = cur_accuracy/self.fp_accuracy
                    if (layer_num+1) % num_layers_together == 0 or layer_num == self.num_layers-1:
                        os.system("python3 compress_classifier.py --arch mobilenet ../../../data.imagenet_100 -p 30 -j=4 --resume ./mobilenet.pth.tar --quantize-eval --compress mobilenet_bn_wrpn.yaml --epochs 1 --lr 0.01")
                        cur_accuracy = float(open("val_accuracy.txt").readlines()[0])
                        self.quant_reward_const = cur_accuracy/self.fp_accuracy
                        print("----------------------------------Cur Accuracy ", cur_accuracy, "--------------------------------")
                        #Use new reward function
                        #reward = self.calculate_network_reward(cur_accuracy, self.fp_accuracy, bitwidth_layers)
                        reward = self.calculate_reward_shaping(cur_accuracy)
                        print("Reward ", reward)
                    else:
                        cur_accuracy = 0
                        reward = 0
                    rewards.append(reward)

                    self.update_quant_state(bitwidth_layers)
                    data = [i, layer_num, self.quant_state, cur_accuracy, reward]
                    data2 = self.Policy.get_action_prob(obs=[s])
                    for each in bitwidth_layers:
                        data.append(each)
                    for each in data2[0]:
                        data.append(each)
                    write_to_csv(data)

                    if (layer_num+1) % num_layers_together == 0 or layer_num == self.num_layers-1:
                        #os.system("python3 compress_classifier.py --arch svhn ../../../data.svhn --quantize-eval --compress ./svhn_bn_dorefa.yaml --epochs 1 --resume ./svhn.pth.tar --lr 0.001")
                        #cur_accuracy = float(open("val_accuracy.txt").readlines()[0])
                        #self.quant_reward_const = cur_accuracy/self.fp_accuracy
                        #Use new reward function
                        #reward = self.calculate_network_reward(cur_accuracy, self.fp_accuracy, bitwidth_layers)
                        #reward = self.calculate_reward_shaping(cur_accuracy)
                        #rewards[-1] = reward
                        v_preds_next = v_preds[1:] + [0]

                        gaes = self.PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)

                        # convert list to numpy array for feeding tf.placeholder
                        observations = np.reshape(s_history, newshape=[-1] + list([7]))
                        actions = np.array(a_history).astype(dtype=np.int32)
                        rewards = np.array(rewards).astype(dtype=np.float32)
                        v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
                        gaes = np.array(gaes).astype(dtype=np.float32)
                        if num_layers_together != 1:
                            gaes = (gaes - gaes.mean()) / gaes.std()
                        
                        self.PPO.assign_policy_parameters()
                        
                        inp = [observations, actions, rewards, v_preds_next, gaes]
                        print(inp)
                        
                        # train
                        for epoch in range(1):
                            sample_indices = np.random.randint(low=0, high=observations.shape[0], size=num_layers_together)  # indices are in [low, high)
                            sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                            print(sampled_inp)
                            self.PPO.train(obs=sampled_inp[0], actions=sampled_inp[1], rewards=sampled_inp[2], v_preds_next=sampled_inp[3], gaes=sampled_inp[4])
                        
                        summary = self.PPO.get_summary(obs=inp[0], actions=inp[1], rewards=inp[2], v_preds_next=inp[3], gaes=inp[4])[0]
                        
                        writer.add_summary(summary, i)
                        writer.close()
                        s_history  = []
                        a_history  = []
                        rewards  = []
                        v_preds = []
                
                    print("End of Episode ", i,", quantized bitwidths ", bitwidth_layers, " Quant_State ", self.quant_state)
                    print("Accuracy with new bit_widths is ", cur_accuracy)

def write_to_csv(step_data):
    with open('releq_svhn_learning_history_log.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(step_data)
# initializing acc_cache dict to use it as global var.
acc_cache = {}
headers = ['episode_num', 'layer_num', 'quant_state', 'acc_state', 'reward',
                        'l1-bits', 'l2-bits', 'l3-bits', 'l4-bits', 'l5-bits', 'l6-bits', 'l7-bits', 'l8-bits', 
                        'prob_2bits','prob_3bits', 'prob_4bits', 'prob_5bits', 'prob_8bits']

with open('releq_svhn_learning_history_log.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(headers)

network_name = "mobilenet"
number_of_layers = 28
layer_info = StringIO("""layer_idx_norm;n;c;k;std
1;32;3;3;0.16152
2;32;1;3;0.35671
3;64;32;1;0.13311
4;64;1;3;0.17684
5;128;64;1;0.09432
6;128;1;3;0.18791
7;128;128;1;0.07951
8;128;1;3;0.11473
9;256;128;1;0.06354
10;256;1;3;0.13977
11;256;256;1;0.05068
12;256;1;3;0.09045
13;512;256;1;0.04042
14;512;1;3;0.12283
15;512;512;1;0.03353
16;512;1;3;0.09269
17;512;512;1;0.03230
18;512;1;3;0.08519
19;512;512;1;0.03307
20;512;1;3;0.07989
21;512;512;1;0.03501
22;512;1;3;0.06508
23;512;512;1;0.03678
24;512;1;3;0.05652
25;1024;512;1;0.02817
26;1024;1;3;0.02874
27;1024;1024;1;0.02032
28;1000;1024;0;0.05217
""")
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
#layer_names = ["features.0", "features.3", "features.7", "features.10", "features.14", "features.17", "features.21", "classifier.0"]
layer_names = ["model.0.0", "model.1.0", "model.1.3", "model.2.0", "model.2.3","model.3.0","model.3.3","model.4.0","model.4.3","model.5.0","model.5.3","model.6.0","model.6.3","model.7.0","model.7.3","model.8.0","model.8.3","model.9.0","model.9.3","model.10.0","model.10.3","model.11.0", "model.11.3", "model.12.0", "model.12.3", "model.13.0", "model.13.3", "fc"]
rl_quant = RLQuantization(number_of_layers, 70, network_name, layer_names, layer_state_info) #num_layers, accuracy, network_name, layer_names, layer_stats
#rl_quant.quantize_layers()
#RL_bw, acc = rl_quant.quantize_layers()
rl_quant.quantize_layers_together(10)
""" finetune stage  """
# start finetuning 
#os.system("python3 compress_classifier.py --arch svhn ../../../data.svhn --quantize-eval --compress ./svhn_bn_dorefa.yaml --epochs 1 --resume ./svhn.pth.tar --lr 0.001")
# print accruacy after finetuning 
#print("RL bitwidth solution:", RL_bw)
#print("Initial accruacy with limited finetuning:", acc)
#cur_accuracy = float(open("val_accuracy.txt").readlines()[0])
#print("Final accruacy after final finetuning:", cur_accuracy)

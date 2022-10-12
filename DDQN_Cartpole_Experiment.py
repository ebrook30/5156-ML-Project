# Ezra Brooks
# ITCS 5156

# Double Deep Q-Learning
# Experiment with CartPole

# Sources Consulted:
# https://blog.paperspace.com/getting-started-with-openai-gym/
# https://github.com/AdalbertoCq/Deep-Learning-Specialization-Coursera
# https://github.com/rlcode/reinforcement-learning
# https://pylessons.com/CartPole-DDQN
# https://pylessons.com/CartPole-reinforcement-learning
# https://pylessons.com/Epsilon-Greedy-DQN
# https://shivam5.github.io/drl/
# https://voyageintech.com/2018/08/14/gym-experiments-cartpole-with-dqn/
# https://www.gocoder.one/blog/rl-tutorial-with-openai-gym

# Additional sources and
# documentation references
# listed where approrpiate

###########
# Imports #
###########

# standard library
import os
import random
from collections import deque

# openAI gym
import gym

# matplotlib + numpy
import pylab
import numpy as np

# tensorflow/keras
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop

#########
# Model #
#########

# model takes predefined inputs and actions
# documented at https://github.com/openai/gym
# the input shape is 4 (position of cart, velocity of cart, angle of pole, rotation rate of pole)
# the action space is 2 (Left and Right)
def basicModel(input_shape, action_space):

    # start with input placeholder defined
    # as a tensor with shape input_shape
    # and set variable X to hold input
    X_input = Input(input_shape)
    X = X_input

    # Define densely-connected neural network layers
    # https://keras.io/api/layers/core_layers/dense/

    # input layer defined by input_shape
    # and first hidden layer with 512 nodes
    X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X)

    # next hidden layer with 256 nodes
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    
    # final hidden layer with 64 nodes
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    # output layer uses the action space to define
    # possible output choices (left/right for cartpole)
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    # group input and output layers into an object
    # https://keras.io/api/models/model/ 
    model = Model(inputs = X_input, outputs = X)

    # compile the model 
    # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop
    model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    # provide summary of model
    model.summary()

    # and return the model
    return model


############
# RL Agent #
############

# agent class defs
class dqnAgent:

    # set up agent parameters
    def __init__(self, envName):

        # set up the gym environment
        self.envName = envName # set the environment name (CartPole-v1) 
        self.env = gym.make(envName) # generate the environment
        self.env.seed(0) # set seed for consistent output 
        self.env._max_episode_steps = 200 # number of steps per episode

        # store the shape space and action count
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        # set number of episodes the agent will run
        self.EPISODES = 500
        self.memory = deque(maxlen=2000)
       
        # initial hyperparameter values
        self.gamma = 0.95  # decay/discount rate
        self.epsilon = 1.0 # exploration rate (rate of random action)
        self.epsilon_min = 0.01 # minimum amount to explore
        self.epsilon_decay = 0.995 # decrease the number of explorations as agent learns

        # additional parameters
        self.batch_size = 32
        self.train_start = 1000
        self.ddqn = True
        self.Soft_Update = False
        self.TAU = 0.1

        # prepare structures and path for outputs
        self.Save_Path = 'Models'
        self.scores, self.episodes, self.average = [], [], []
       
        # print outputs for diagnostics 
        if self.ddqn:
            print("----------Double DQN--------")
            self.Model_name = os.path.join(self.Save_Path,"DDQN_"+self.envName+".h5")
        else:
            print("-------------DQN------------")
            self.Model_name = os.path.join(self.Save_Path,"DQN_"+self.envName+".h5")
        
        # create main model
        self.model = basicModel(input_shape=(self.state_size,), action_space = self.action_size)
        self.target_model = basicModel(input_shape=(self.state_size,), action_space = self.action_size)


    # Update the target model after a given time interval
    def update_target_model(self):
        if not self.Soft_Update and self.ddqn:
            self.target_model.set_weights(self.model.get_weights())
            return
        if self.Soft_Update and self.ddqn:
            q_model_theta = self.model.get_weights()
            target_model_theta = self.target_model.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1-self.TAU) + q_weight * self.TAU
                target_model_theta[counter] = target_weight
                counter += 1
            self.target_model.set_weights(target_model_theta)

    # store states, actions, and resulting rewards to the memory list
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    # get the next action
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    # train the network using
    # elements from memory
    def replay(self):

        if len(self.memory) < self.train_start:
            return

        # Randomly sample minibatch from memory
        minibatch = random.sample(self.memory, min(self.batch_size, self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # calculate values for mini batch
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # perform batch prediction
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)
        target_val = self.target_model.predict(next_state)

        for i in range(len(minibatch)):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                if self.ddqn: # Double - DQN
                    # current Q Network selects the action
                    # a'_max = argmax_a' Q(s', a')
                    a = np.argmax(target_next[i])
                    # target Q Network evaluates the action
                    # Q_max = Q_target(s', a'_max)
                    target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])   
                else: # Standard - DQN
                    # DQN chooses the max Q value among next actions
                    # selection and evaluation of action is on the target Q Network
                    # Q_max = max_a' Q_target(s', a')
                    target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)


    # load a pre-trained model
    def load(self, name):
        self.model = load_model(name)

    # save a model for future use
    def save(self, name):
        self.model.save(name)

    # plot outputs
    pylab.figure(figsize=(18, 9))
    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores) / len(self.scores))
        pylab.plot(self.episodes, self.average, 'r')
        pylab.plot(self.episodes, self.scores, 'b')
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Steps', fontsize=18)
        dqn = 'DQN_'
        softupdate = ''
        if self.ddqn:
            dqn = 'DDQN_'
        if self.Soft_Update:
            softupdate = '_soft'
        try:
            pylab.savefig(dqn+self.envName+softupdate+".png")
        except OSError:
            pass

        return str(self.average[-1])[:5]
    
    # run the agent
    def run(self):
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                #self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:
                    # every step update target model
                    self.update_target_model()
                    
                    # every episode, plot the result
                    average = self.PlotModel(i, e)
                     
                    print("episode: {}/{}, score: {}, e: {:.2}, average: {}".format(e, self.EPISODES, i, self.epsilon, average))
                    if i == self.env._max_episode_steps:
                        print("Saving trained model as cartpole-ddqn.h5")
                        #self.save("cartpole-ddqn.h5")
                        break
                self.replay()


    # test the model
    def test(self):

        # load stored model
        self.load("cartpole-ddqn.h5")

        # iterate over episodes
        for e in range(self.EPISODES):
            # prepare state and wait until process is complete
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            # run test
            while not done:
                # render the environment
                self.env.render()
                # compute the action based on predicted state
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                    break


if __name__ == "__main__":

    # set OpenAI gym environment
    # to use (cartpole-v1)
    envName = 'CartPole-v1'

    # prepare the agent object instance
    agent = dqnAgent(envName)

    # run the agent
    agent.run()

    # uncomment to test the agent
    # against pretrained model
    # agent.test()

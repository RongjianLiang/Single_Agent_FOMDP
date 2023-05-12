import tensorflow as tf
# import keras
# from keras.models import Sequential, Model, load_model
# from keras.layers import Dense, Dropout, Conv3D, MaxPooling3D, Activation, Flatten, concatenate, Input
# from keras.callbacks import TensorBoard
# from keras.utils.vis_utils import plot_model
# from tensorflow.keras.optimizers import Adam
import csv
# import statistics
import seaborn as sns
# from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import random
import os
# import time
# from datetime import datetime
# import math
from tqdm import tqdm
# from PIL import Image

# import modifiedTB as tb
from agent import Point
from TrainingEnv import TrainingEnv
from DQNAgent import DQNAgent
from TestEnv import TestEnv

episodes = 2500
action_space_size = 6
epsilon = 0.5
size = 10  # this should be the same as in config file
goal_reward = 1.00
AGGREGATE_STATS_EVERY = 10
SAVE_MODEL_EVERY = 20
MIN_EPSILON = 0.001
EPSILON_DECAY = 0.9995
MODEL_NAME = 'Single_Agent_FOMDP'


def main():
    global epsilon
    env = TrainingEnv()
    ep_rewards = [-1]

    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    if not os.path.isdir('models'):
        os.makedirs('models')

    tf.autograph.set_verbosity(0)
    agent = DQNAgent()

    for episode in tqdm(range(1, episodes + 1), ascii=True, unit="episode"):
        agent.tensorboard.step = episode

        episode_reward = 0
        step = 1
        current_state = env.reset()
        min_reward = []
        max_reward = []
        average_reward = []

        done = False
        # print("Running episodes: {}".format(episode))
        while not done:
            # print("--Stepping: {}".format(step))

            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs([np.array([current_state[:size ** 3].reshape((size, size, size))]),
                                                 np.array([current_state[size ** 3:]])]))
            else:
                action = np.random.randint(0, action_space_size)

            new_state, reward, done = env.step(action)

            episode_reward += reward

            agent.update_replay_memory((current_state, action, reward, new_state, done))

            # Hindsight Experience Replay
            if not done:
                new_reward = Point(*env.agent[0].location())
                new_reward.action(action)
                if not env.agent[0] == new_reward:
                    HER_state = current_state.copy()
                    HER_state[-3:] = env.agent[0].vector(new_reward)
                    HER_new_state = new_state.copy()
                    HER_new_state[-3:] = np.array([0, 0, 0])
                    agent.update_replay_memory((HER_state, action, goal_reward, HER_new_state, True))

            agent.train(done, step)
            current_state = new_state
            step += 1

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=epsilon)

        if not episode % SAVE_MODEL_EVERY and episode != 0:
            # Save model
            agent.model.save(
                f'models/{MODEL_NAME}_{str(episode).zfill(5)}_{max_reward:_>7.2f}'
                f'max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min.model')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)


def logging():
    log = [("1", 10, 10, 0), ("2", 10, 10, 0), ("3", 10, 10, 10),
           ("4", 10, 10, 10), ("5", 10, 25, 0), ("6", 10, 25, 0),
           ("7", 10, 25, 10), ("8", 10, 25, 10), ("9", 20, 50, 25),
           ("10", 20, 50, 25), ("11", 20, 100, 25), ("12", 20, 100, 25)]

    model = [
        "1_05000____0.90max___-0.76avg___-1.29min.model",
        "2_00200____0.96max___-0.32avg___-0.84min.model",
        "3_00600____0.48max___-0.63avg___-1.39min.model",
        "4_10200____0.98max____0.77avg___-0.39min.model",
        "5_00400___-0.25max___-0.66avg___-1.21min.model",
        "6_00200____1.00max___-0.10avg___-0.63min.model",
        "7_00200____0.96max____0.10avg___-0.59min.model",
        "8_11400____0.99max____0.78avg___-0.37min.model",
        "9_01800___-0.25max___-0.55avg___-1.09min.model",
        "10_05000___-0.27max___-0.47avg___-0.79min.model",
        "11_08200___-0.25max___-0.51avg___-1.04min.model",
        "12_08600____0.94max____0.34avg___-0.60min.model"
    ]

    for idx, i in enumerate(log):

        MODEL_NAME, SIZE, N_BUILDINGS, N_OBSTACLES = i

        env = TestEnv()
        agent = DQNAgent("models/" + model[idx])

        N_TESTS = 1

        rewards = []
        lengths = []

        for ep in range(N_TESTS):

            episode_reward = 0
            step = 1

            env.reset()
            L1 = sum(abs(env.agent[0].location() - env.agent[1].location()))
            while L1 != SIZE:
                env.reset()
                L1 = sum(abs(env.agent[0].location() - env.agent[1].location()))

            done = False

            while not done:
                env.render(save=f"visualisations/{MODEL_NAME}_{step}.png")
                reward, done = env.step()
                episode_reward += reward
                step += 1
            env.render(save=f"visualisations/{MODEL_NAME}_{step}.png")

            print(f"Model {MODEL_NAME}, Reward {episode_reward}, Length {step}")

    #         rewards.append(episode_reward)
    #         lengths.append(step)
    #         if ep % 20 == 0:
    #             print('.', end="")
    #             if ep % 100 == 0:
    #                 print()

    #     with open(f'{MODEL_NAME}_test.csv', 'w', newline='') as file:
    #         mywriter = csv.writer(file, delimiter=',')
    #         for i in range(len(rewards)):
    #             mywriter.writerow((rewards[i], lengths[i]))

    #     avg_distance = statistics.mean([l for l in lengths if l != 101]) * DELTA_X_M
    #     standard_dev = statistics.stdev([l for l in lengths if l != 101]) * DELTA_X_M
    #     perc_90 = np.percentile([l for l in lengths if l != 101], 90)
    #     perc_50 = np.percentile([l for l in lengths if l != 101], 50)

    #     sns.set(palette="flare")
    #     plt.hist([r for r in rewards if r > 0], density=True, bins=50)
    #     plt.ylabel('Freq')
    #     plt.xlabel('Reward')
    #     plt.title(f"Run {MODEL_NAME}: Avg Reward = {sum(rewards)/len(rewards):.3f}")
    #     plt.savefig(f"{MODEL_NAME}_reward.png", dpi=960)
    #     plt.show()


def visualisation():
    labels = [t / 5 for t in range(10, 30, 2)]
    sns.set(palette="dark")
    plt.figure(figsize=(6, 7.5))

    m = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')

    for i in range(1, 9):

        rewards = []
        lengths = []
        with open(f'{i}_test.csv') as file:
            myreader = csv.reader(file)
            for j in myreader:
                rewards.append(float(j[0]))
                lengths.append(float(j[1]))

        values = [len([l for l in lengths if l == j or l == j + 1]) / 1000 for j in range(11, 31, 2)]
        plt.plot(labels, values, '-' + m[i - 1], label=f"Run {i}")

    plt.ylabel('Frequency')
    plt.ylim([-0.05, 0.55])
    plt.xlabel('Travel Time (s)')
    plt.title(f"Travel Time Distributions, FOMDP Simulation Run 1 - 8")
    plt.legend()

    plt.savefig(f"Combined_time_1.png", dpi=480)
    plt.show()

    labels = [t / 5 for t in range(20, 40, 2)]
    plt.figure(figsize=(6, 7.5))

    for i in range(9, 13):

        rewards = []
        lengths = []
        with open(f'{i}_test.csv') as file:
            myreader = csv.reader(file)
            for j in myreader:
                rewards.append(float(j[0]))
                lengths.append(float(j[1]))

        values = [len([l for l in lengths if l == j or l == j + 1]) / 1000 for j in range(21, 41, 2)]
        plt.plot(labels, values, '-' + m[i - 9], label=f"Run {i}")

    plt.ylabel('Frequency')
    plt.ylim([-0.05, 0.55])
    plt.xlabel('Travel Time (s)')
    plt.title(f"Travel Time Distributions, FOMDP Simulation Run 9 - 12")
    plt.legend()

    plt.savefig(f"Combined_time_2.png", dpi=480)
    plt.show()


if __name__ == "__main__":
    main()
    logging()
    visualisation()

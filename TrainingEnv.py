# import configparser
# import time

from agent import Point
import numpy as np
import matplotlib.pyplot as plt
import math
import random


class TrainingEnv:
    def __init__(self, config=None):
        if config is not None:
            self.params_from_config(config)
        else:
            self.init_params()
        # TODO: define other parameters below
        # TODO: implement the reset function, which re-generate the env
        self.reset()

    def init_params(self):
        self.episode_step = 0
        self.agent_path = []
        self.agent = []
        # environment params
        self.env_size = 10
        self.n_obstacles = 10
        self.n_buildings = 10
        self.max_height = 8
        self.collision_penalty = 0.25
        self.move_penalty = 0.01
        self.ground_prox_penalty = 0.2
        self.goal_reward = 1.00
        self.hetero_reward = False
        self.binary_envs = True

    def params_from_config(self, config):
        self.env_size = config.getint('SIZE', 10)
        self.n_buildings = config.getint('N_BUILDINGS', 10)
        self.n_obstacles = config.getint('N_OBSTACLES', 10)
        self.max_height = config.getint('MAX_HEIGHT', 8)
        self.move_penalty = config.getfloat('MOVE_PENALTY', 0.01)
        self.collision_penalty = config.getfloat('COLLISION_PENALTY', 0.25)
        self.ground_prox_penalty = config.getfloat('GROUND_PROX_PENALTY', 0.2)
        self.goal_reward = config.getfloat('GOAL_REWARD', 1.00)
        self.binary_envs = config.getbool('BINARY_ENV', False)
        self.hetero_reward = config.getbool('HETERO_REWARD', True)

    def reset(self):

        self.terrain_map = self.generate_terrain()  # TODO: shall this be refactored?

        empty_blocks_iter = self.empty_blocks(self.terrain_map)
        self.agent = [Point(*next(empty_blocks_iter)), Point(*next(empty_blocks_iter))]
        self.drones_map = np.zeros((self.env_size, self.env_size, self.env_size))
        self.goals_map = np.zeros((self.env_size, self.env_size, self.env_size))
        self.drones_map[tuple(self.agent[0].location())] = 1
        self.goals_map[tuple(self.agent[1].location())] = 1
        self.dynamic_obs = [Point(*next(empty_blocks_iter)) for i in range(self.n_obstacles)]
        self.obstacles_map = np.zeros((self.env_size, self.env_size, self.env_size))
        for obs in self.dynamic_obs:
            self.obstacles_map[tuple(obs.location())] = 1

        # Keep track of the drone's trajectory
        self.agent_path = [self.agent[0].location()]

        self.episode_step = 0

        return self.generate_state()

    def step(self, action):
        self.episode_step += 1
        done = False

        self.drones_map[tuple(self.agent[0].location())] = 0
        self.agent[0].action(action).within_bounds(self.env_size, self.env_size, self.env_size)
        self.drones_map[tuple(self.agent[0].location())] = 1

        self.agent_path.append(self.agent[0].location())

        terrain = self.terrain_map == 1
        obstacles = self.obstacles_map == 1
        drones = self.drones_map == 1

        if (terrain | obstacles).astype(int)[tuple(self.agent[0].location())] == 1:
            reward = -self.collision_penalty
            done = True
        elif self.agent[0] == self.agent[1]:
            reward = self.goal_reward
            done = True
        else:
            reward = (-self.move_penalty) * \
                     (1.0 + (self.hetero_reward * 2.0 * self.terrain_map[tuple(self.agent[0].location())]))

        for obs in self.dynamic_obs:
            self.obstacles_map[tuple(obs.location())] = 0
            obstacles = self.obstacles_map == 1
            while True:
                if not obs.drift().within_bounds(self.env_size, self.env_size, self.env_size):
                    obs.drift_heading = 2 * np.random.rand(3) - 1
                if (terrain | obstacles | drones).astype(int)[tuple(obs.location())] != 1:
                    break
            self.obstacles_map[tuple(obs.location())] = 1
            obstacles = self.obstacles_map == 1

        new_observation = self.generate_state()

        if self.episode_step >= 50:
            done = True

        return new_observation, reward, done

    def render(self, elev=60, azim=45, save=""):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True)

        terrain = self.terrain_map == 1
        drones = self.drones_map == 1
        goals = self.goals_map == 1
        obstacles = self.obstacles_map == 1

        voxelarr = terrain | drones | goals | obstacles
        colors = np.empty(terrain.shape, dtype=object)
        colors[terrain] = '#7A88CCC0'
        colors[drones] = '#FFD65DC0'
        colors[goals] = '#607D3BC0'
        colors[obstacles] = '#FDA4BAC0'
        ax.voxels(voxelarr, facecolors=colors, shade=True)

        #         for action in range(len(self.path)-1):
        #             xline = np.linspace(self.path[action][0] + 0.5, self.path[action+1][0] + 0.5, 1000)
        #             yline = np.linspace(self.path[action][1] + 0.5, self.path[action+1][1] + 0.5, 1000)
        #             zline = np.linspace(self.path[action][2] + 0.5, self.path[action+1][2] + 0.5, 1000)
        #             ax.plot3D(xline, yline, zline, 'black')

        if save != "":
            plt.savefig(save)
        plt.show()

    def generate_terrain(self):
        terrain = np.zeros((self.env_size, self.env_size, self.env_size))

        if self.ground_prox_penalty and not self.binary_envs:
            for i in range(self.env_size // 2):
                terrain[:, :, i] = self.ground_prox_penalty * (self.env_size // 2 - i) / (self.env_size // 2)

        for i in range(self.n_buildings):
            while True:
                # Generate random numbers in intervals of 0.5
                x, y = np.random.randint(0, self.env_size * 2, 2) / 2
                # Generate random building height
                z = np.random.randint(0, self.max_height)
                # Check if existing buildings exist. If so, regenerate. Otherwise, keep building.
                if np.all(terrain[math.floor(x):math.ceil(x) + 1, math.floor(y):math.ceil(y) + 1, 0:z] != 1):
                    if not self.binary_envs:
                        terrain[math.floor(x) - 1:math.ceil(x) + 2, math.floor(y) - 1:math.ceil(y) + 2, 0:z + 1] \
                            = terrain[math.floor(x) - 1:math.ceil(x) + 2, math.floor(y) - 1:math.ceil(y) + 2, 0:z + 1]. \
                            clip(min=0.5)
                    terrain[math.floor(x):math.ceil(x) + 1, math.floor(y):math.ceil(y) + 1, 0:z].fill(1)
                    break
        return terrain

    def generate_state(self):
        return np.append(np.maximum(self.terrain_map, self.obstacles_map), self.agent[0].vector(self.agent[1]))

    def empty_blocks(self, occupied):
        empty_blocks = [[x, y, z] for x in range(self.env_size) for y in range(self.env_size) for z in
                        range(self.env_size) if
                        occupied[x, y, z] != 1]
        random.shuffle(empty_blocks)
        return iter(empty_blocks)


"""
if __name__ == "__main__":

    config_file = 'params.cfg'
    config_parser = configparser.ConfigParser()
    config_parser.read(config_file)
    env = TrainingEnv(config_file)

    while True:
        t0 = time.time()
        env.render()
        # TODO: check if the env episode is finished
        # TODO: close env and print finish message
        # TODO: else, update the state, reward, and print step message
        # TODO: perform a finish check
"""

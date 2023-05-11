class TestEnv:
    def __init__(self):
        self.terrain_map = None
        self.episode_step = None
        self.path = None
        self.obstacles_map = None
        self.dynamic_obs = None
        self.goals_map = None
        self.drones_map = None
        self.agent = None

    def reset(self):
        self.terrain_map = self.generate_terrain()

        empty_blocks_iter = self.empty_blocks(self.terrain_map)
        self.agent = [Point(*next(empty_blocks_iter)), Point(*next(empty_blocks_iter))]
        self.drones_map = np.zeros((SIZE, SIZE, SIZE))
        self.goals_map = np.zeros((SIZE, SIZE, SIZE))
        self.drones_map[tuple(self.agent[0].location())] = 1
        self.goals_map[tuple(self.agent[1].location())] = 1
        self.dynamic_obs = [Point(*next(empty_blocks_iter)) for i in range(N_OBSTACLES)]
        self.obstacles_map = np.zeros((SIZE, SIZE, SIZE))
        for obs in self.dynamic_obs:
            self.obstacles_map[tuple(obs.location())] = 1

        # Keep track of the drone's trajectory
        self.path = [self.agent[0].location()]

        self.episode_step = 0

        # return self.generate_state()

    def step(self):
        self.episode_step += 1
        done = False
        action = 0

        terrain = self.terrain_map == 1
        obstacles = self.obstacles_map == 1
        drones = self.drones_map == 1
        goals = self.goals_map == 1

        if np.random.rand() < 0.95:
            for i in range(ACTION_SPACE_SIZE):
                current_state = self.generate_state()
                action = np.argsort(agent.get_qs([np.array([current_state[:SIZE ** 3].reshape((SIZE, SIZE, SIZE))]),
                                                  np.array([current_state[SIZE ** 3:]])]))[-i - 1]
                #                 action = np.argsort(agent.get_qs([np.array([current_state[:SIZE**3]]).
                #                 reshape((SIZE,SIZE,SIZE)), np.array([current_state[SIZE**3:]])]))[-i-1]
                n = self.agent[0].copy().action(action)
                n.within_bounds(SIZE, SIZE, SIZE)
                if (terrain | obstacles | drones).astype(int)[tuple(n.location())] != 1:
                    break
        else:
            for action in np.random.permutation(ACTION_SPACE_SIZE):
                n = self.agent[0].copy().action(action)
                n.within_bounds(SIZE, SIZE, SIZE)
                if (terrain | obstacles | drones).astype(int)[tuple(n.location())] != 1:
                    break

        self.drones_map[tuple(self.agent[0].location())] = 0
        self.agent[0].action(action).within_bounds(SIZE, SIZE, SIZE)
        self.drones_map[tuple(self.agent[0].location())] = 1

        drones = self.drones_map == 1

        self.path.append(self.agent[0].location())

        #         if (terrain|obstacles).astype(int)[tuple(self.agent[0].location())] == 1:
        #             reward = -COLLISION_PENALTY
        #             done = True
        if self.agent[0] == self.agent[1]:
            reward = GOAL_REWARD
            done = True
        else:
            reward = (-MOVE_PENALTY) * (1.0 + (HETERO_REWARD * 2.0 * self.terrain_map[tuple(self.agent[0].location())]))

        for obs in self.dynamic_obs:
            while True:
                n = obs.copy()
                n.drift_heading = obs.drift_heading
                if n.drift().within_bounds(SIZE, SIZE, SIZE):
                    if (terrain | obstacles | drones | goals).astype(int)[tuple(n.location())] != 1:
                        break
                else:
                    obs.drift_heading = 2 * np.random.rand(3) - 1
                    continue
            self.obstacles_map[tuple(obs.location())] = 0
            obs.x, obs.y, obs.z = n.x, n.y, n.z
            self.obstacles_map[tuple(obs.location())] = 1
            obstacles = self.obstacles_map == 1

        if self.episode_step >= 100:
            done = True

        return reward, done

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
        terrain = np.zeros((SIZE, SIZE, SIZE))

        if GROUND_PROX_PENALTY and not BINARY_ENV:
            for i in range(SIZE // 2):
                terrain[:, :, i] = GROUND_PROX_PENALTY * (SIZE // 2 - i) / (SIZE // 2)

        for i in range(N_BUILDINGS):
            while True:
                # Generate random numbers in intervals of 0.5
                x, y = np.random.randint(0, SIZE * 2, 2) / 2
                # Generate random building height
                z = np.random.randint(0, MAX_HEIGHT)
                # Check if existing buildings exist. If so, regenerate. Otherwise, keep building.
                if np.all(terrain[math.floor(x):math.ceil(x) + 1, math.floor(y):math.ceil(y) + 1, 0:z] != 1):
                    if not BINARY_ENV:
                        terrain[math.floor(x) - 1:math.ceil(x) + 2, math.floor(y) - 1:math.ceil(y) + 2,
                        0:z + 1] = terrain[math.floor(x) - 1:math.ceil(x) + 2, math.floor(y) - 1:math.ceil(y) + 2,
                                   0:z + 1].clip(min=0.5)
                    terrain[math.floor(x):math.ceil(x) + 1, math.floor(y):math.ceil(y) + 1, 0:z].fill(1)
                    break
        return terrain

    def generate_state(self):
        return np.append(np.maximum(self.terrain_map, self.obstacles_map), self.agent[0].vector(self.agent[1]))

    def empty_blocks(self, occupied):
        empty_blocks = [[x, y, z] for x in range(SIZE) for y in range(SIZE) for z in range(SIZE) if
                        occupied[x, y, z] != 1]
        random.shuffle(empty_blocks)
        return iter(empty_blocks)
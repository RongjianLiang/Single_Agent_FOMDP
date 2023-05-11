import tensorflow as tf
from tensorflow import keras
import random
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Conv3D, MaxPooling3D, Flatten, concatenate, Input
from tensorflow.keras.optimizers import Adam
import modifiedTB as tb
from collections import deque
from datetime import datetime


class DQNAgent(object):
    def __init__(self, filepath="", config=None):
        self.min_replay_memory_size = None
        self.mini_batch_size = None
        self.model_size = None
        self.action_space_size = None
        self.replay_memory_size = None
        self.model_name = 'Single_Agent_FOMDP'
        self.learning_rate = None
        self.discount = None
        self.TAU = None

        if config is not None:
            self.params_from_config(config)
        else:
            self.init_params()

        if filepath == "":
            # Create primary model and target model
            self.model = self.create_model(self.model_size, 3, self.action_space_size)
            self.target_model = self.create_model(self.model_size, 3, self.action_space_size)
            self.target_model.set_weights(self.model.get_weights())
        else:
            # Load architecture and weights from model file
            self.model = keras.models.load_model(filepath)
            self.target_model = keras.models.load_model(filepath)

        self.replay_memory = deque(maxlen=self.replay_memory_size)

        cur_time = datetime.now().strftime("%m-%d_%H%M")
        self.tensorboard = tb.ModifiedTensorBoard(log_dir=f"logs/{self.model_name}")
        self.target_update_counter = 0

    def init_params(self):
        self.min_replay_memory_size = 100
        self.mini_batch_size = 100
        self.model_size = 10
        self.action_space_size = 6
        self.replay_memory_size = 100
        self.learning_rate = 0.001
        self.discount = 0.95
        self.TAU = 0.97

    def params_from_config(self, config):
        self.min_replay_memory_size = config.getint('MIN_REPLAY_MEMORY_SIZE', 100)
        self.replay_memory_size = config.getint('REPLAY_MEMORY_SIZE', 100)
        self.model_size = config.getint('SIZE', 10)
        self.action_space_size = config.getint('ACTION_SPACE_SIZE', 6)
        self.discount = config.getfloat('DISCOUNT', 0.95)
        self.mini_batch_size = config.getint('MINI_BATCH_SIZE', 32)
        self.learning_rate = config.getfloat('LEARNING_RATE', 0.001)
        self.TAU = config.getfloat('TAU', 0.97)

    def create_model(self, terrain_dim, goal_dim, output_dim):

        #         terrain_in = Input(shape=(terrain_dim,terrain_dim,terrain_dim,1))
        #         x = Conv3D(128, (2,2,2), activation="relu", padding='same')(terrain_in)
        #         x = Conv3D(128, (2,2,2), activation="relu", padding='same')(x)
        #         x = Conv3D(128, (2,2,2), activation="relu", padding='same')(x)
        #         x = MaxPooling3D((2,2,2))(x)
        #         x = Dropout(0.1)(x)

        #         x = Conv3D(256, (2,2,2), activation="relu", padding='same')(x)
        #         x = Conv3D(256, (2,2,2), activation="relu", padding='same')(x)
        #         x = Conv3D(256, (2,2,2), activation="relu", padding='same')(x)

        #         terrain_out = Flatten()(x)

        #         goal_in = Input(shape=(goal_dim))
        #         goal_out = (Dense(64, activation='relu'))(goal_in)

        #         concat = concatenate([terrain_out, goal_out])
        #         x = (Dense(256, activation='relu'))(concat)
        #         x = (Dense(256, activation='relu'))(x)
        #         x = (Dense(128, activation='relu'))(x)
        #         model_out = (Dense(output_dim, activation='linear'))(x)

        #         model = Model([terrain_in, goal_in], model_out)

        #         model.compile(loss=tf.keras.losses.Huber(), optimizer=Adam(learning_rate=LEARNING_RATE))

        #         return model

        terrain_in = Input(shape=(terrain_dim, terrain_dim, terrain_dim, 1))

        x = Conv3D(32, (2, 2, 2), activation="relu")(terrain_in)
        x = Conv3D(32, (2, 2, 2), activation="relu")(x)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        x = Dropout(0.1)(x)

        x = Conv3D(64, (2, 2, 2), activation="relu")(x)
        x = Conv3D(64, (2, 2, 2), activation="relu")(x)
        terrain_out = Flatten()(x)

        goal_in = Input(shape=goal_dim)
        goal_out = (Dense(64, activation='relu'))(goal_in)

        concat = concatenate([terrain_out, goal_out])
        x = (Dense(128, activation='relu'))(concat)
        x = (Dense(128, activation='relu'))(x)
        model_out = (Dense(output_dim, activation='linear'))(x)

        model = Model([terrain_in, goal_in], model_out)

        model.compile(loss=tf.keras.losses.Huber(), optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        with tf.device('/GPU:0'):
            return self.model.predict(state)[0]

    def train(self, terminal_state, step):
        # Only train if there's sufficient replay memory size
        if len(self.replay_memory) >= self.min_replay_memory_size:
            minibatch = random.sample(self.replay_memory, self.mini_batch_size)

            current_states = [np.array([m[0][:self.model_size ** 3].
                                       reshape((self.model_size, self.model_size, self.model_size)) for m in
                                        minibatch]),
                              np.array([m[0][self.model_size ** 3:] for m in minibatch])]
            with tf.device('/GPU:0'):
                current_qs_list = self.model.predict(current_states)
            new_states = [np.array([m[3][:self.model_size ** 3].
                                   reshape((self.model_size, self.model_size, self.model_size)) for m in minibatch]),
                          np.array([m[3][self.model_size ** 3:] for m in minibatch])]
            with tf.device('/GPU:0'):
                future_qs_list = self.model.predict(new_states)
                future_qs_target_list = self.target_model.predict(new_states)

            X = current_states
            Y = []

            for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
                if not done:
                    future_q = future_qs_target_list[index][np.argmax(future_qs_list[index])]
                    new_q = reward + self.discount * future_q
                else:
                    new_q = reward

                current_qs = current_qs_list[index]
                current_qs[action] = new_q

                Y.append(current_qs)

            with tf.device('/GPU:0'):
                self.model.fit(X, [np.array(Y)], batch_size=self.mini_batch_size, verbose=0, shuffle=False,
                               callbacks=[self.tensorboard] if terminal_state else None)

            for t, e in zip(self.target_model.trainable_variables, self.model.trainable_variables):
                t.assign(t * self.TAU + (1 - self.TAU) * e)

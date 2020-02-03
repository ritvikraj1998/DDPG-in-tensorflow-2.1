import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
tf.keras.backend.set_floatx('float64')

"""
Critic model
lr = 0.001
l2 weight decay 0.01
gamma = 0.99 for loss
relu,relu,relu
400,300,4
action added in dense2


"""


class Critic:
    def __init__(self, state_size, action_dim, tau):
        self.tau = tau
        self.state_size = state_size
        self.action_dim = action_dim
        self.critic = self.build_network()
        self.critic.compile(Adam(learning_rate=0.00001), loss='mse')
        self.target_critic = self.build_network()
        self.target_critic.compile(Adam(learning_rate=0.00001), loss='mse')

    def build_network(self):
        state_input = tf.keras.Input(shape=self.state_size)
        action_input = tf.keras.Input(shape=self.action_dim)
        dense1 = Dense(units=40, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01),
                       kernel_initializer=tf.random_uniform_initializer(minval=1 / np.sqrt(40),
                                                                        maxval=1 / np.sqrt(40)))(state_input)
        state_action_input = tf.keras.layers.concatenate([dense1, action_input])
        dense2 = Dense(units=30, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01),
                       kernel_initializer=tf.random_uniform_initializer(minval=-1 / np.sqrt(30),
                                                                        maxval=1 / np.sqrt(30)))(state_action_input)
        output_layer = Dense(units=1, activation='linear',
                             kernel_regularizer=tf.keras.regularizers.l2(0.01),
                             kernel_initializer=tf.random_uniform_initializer(minval=-3 * 0.001, maxval=3 * 0.001))(
            dense2)
        return tf.keras.models.Model(inputs=[state_input, action_input], outputs=[output_layer])

    def update_target(self):
        w, target_w = self.critic.get_weights(), self.target_critic.get_weights()
        for i in range(len(w)):
            target_w[i] = self.tau * w[i] + (1 - self.tau) * target_w[i]
        self.target_critic.set_weights(target_w)


"""
Actor
lr = 0.0001
relu,relu,tanh
400,300
grad_ys is only needed for advanced use cases. Here is how you can think about it.
tf.gradients allows you to compute tf.gradients(y, x, grad_ys) = grad_ys * dy/dx
"""


class Actor:
    def __init__(self, state_size, action_dim, tau):
        self.state_size = state_size
        self.action_dim = action_dim
        self.tau = tau
        self.actor = self.build_network()
        self.target_actor = self.build_network()
        self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)

    def train(self, critic, states):
        states = tf.convert_to_tensor(states, dtype=tf.float64)
        with tf.GradientTape() as tape:
            actions = self.actor(states)
            output = critic.critic([states, actions])
            actor_loss = - tf.reduce_mean(output)
        gradients = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.adam_optimizer.apply_gradients(zip(gradients, self.actor.trainable_weights))

    def build_network(self):
        model = Sequential([
            Dense(units=40, input_dim=self.state_size, activation='relu',
                  kernel_initializer=tf.random_uniform_initializer(minval=1 / np.sqrt(40),
                                                                   maxval=1 / np.sqrt(40))),
            Dense(units=30, activation='relu', kernel_initializer=tf.random_uniform_initializer(minval=1 / np.sqrt(30),
                                                                                                maxval=1 / np.sqrt(
                                                                                                    30))),
            Dense(units=self.action_dim, activation='tanh',
                  kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003))])
        return model

    def update_target(self):
        w, target_w = self.actor.get_weights(), self.target_actor.get_weights()
        for i in range(len(w)):
            target_w[i] = self.tau * w[i] + (1 - self.tau) * target_w[i]
        self.target_actor.set_weights(target_w)


class Noise:
    def __init__(self, mu, sigma=0.2, theta=0.5, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.sigma = sigma
        self.x0 = x0
        self.reset(x0)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self, x0=None):
        self.x_prev = self.x0 if x0 is not None else np.zeros_like(self.mu)


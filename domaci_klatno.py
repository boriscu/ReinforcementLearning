from typing import Tuple
from math import sin, cos
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from tqdm import tqdm


def F_x(
    z: Tuple[float, float, float, float],
    f: float,
    m: float,
    M: float,
    l: float,
    g: float,
    k: float,
) -> float:
    """
    Calculate the acceleration of the cart (x'').

    Args:
    z (Tuple[float, float, float, float]): The state of the system (cart position, cart velocity, pendulum angle, pendulum angular velocity).
    f (float): The force applied to the cart.
    m (float): The mass of the pendulum.
    M (float): The mass of the cart.
    l (float): The length of the pendulum.
    g (float): The acceleration due to gravity.
    k (float): The damping factor.

    Returns:
    float: The acceleration of the cart.
    """
    x, x_dot, theta, theta_dot = z
    numerator = m * g * sin(theta) * cos(theta) - (1 + k) * (
        f + m * l * theta_dot**2 * sin(theta)
    )
    denominator = m * cos(theta) ** 2 - (1 + k) * M
    return numerator / denominator


def F_theta(
    z: Tuple[float, float, float, float],
    f: float,
    m: float,
    M: float,
    l: float,
    g: float,
    k: float,
) -> float:
    """
    Calculate the angular acceleration of the pendulum (theta'').

    Args:
    z (Tuple[float, float, float, float]): The state of the system (cart position, cart velocity, pendulum angle, pendulum angular velocity).
    f (float): The force applied to the cart.
    m (float): The mass of the pendulum.
    M (float): The mass of the cart.
    l (float): The length of the pendulum.
    g (float): The acceleration due to gravity.
    k (float): The damping factor.

    Returns:
    float: The angular acceleration of the pendulum.
    """
    x, x_dot, theta, theta_dot = z
    numerator = M * g * sin(theta) - cos(theta) * (
        f + m * l * theta_dot**2 * sin(theta)
    )
    denominator = (1 + k) * M * l - m * l * cos(theta) ** 2
    return numerator / denominator


def update_state(
    z: Tuple[float, float, float, float],
    f: float,
    T: float,
    m: float,
    M: float,
    l: float,
    g: float,
    k: float,
) -> Tuple[float, float, float, float]:
    """
    Update the state of the system using Euler's method.

    Args:
    z (Tuple[float, float, float, float]): The current state of the system (z1: x, z2: x_dot, z3: theta, z4: theta_dot).
    f (float): The force applied to the cart.
    T (float): The time step size.
    m (float): The mass of the pendulum.
    M (float): The mass of the cart.
    l (float): The length of the pendulum.
    g (float): The acceleration due to gravity.
    k (float): The damping factor.

    Returns:
    Tuple[float, float, float, float]: The updated state of the system.
    """
    z1, z2, z3, z4 = z
    z1 += T * z2
    z2 += T * F_x((z1, z2, z3, z4), f, m, M, l, g, k)
    z3 += T * z4
    z4 += T * F_theta((z1, z2, z3, z4), f, m, M, l, g, k)
    return z1, z2, z3, z4


class InvertedPendulumEnv:
    def __init__(
        self,
        max_force: float,
        time_limit: int,
        m: float,
        M: float,
        l: float,
        cart_boundary: float = 2.0,
        g: float = 9.81,
        k: float = 0.0,
        T: float = 0.1,
        epsilon: float = 0.1,
        alpha: float = 0.1,
        gamma: float = 0.9,
        resolution: int = 10,
    ):
        """
        Initialize the simulation environment for the inverted pendulum.

        Args:
        max_force (float): Maximum magnitude of force that can be applied.
        time_limit (int): The maximum number of time steps in an episode.
        m (float): The mass of the pendulum.
        M (float): The mass of the cart.
        l (float): The length of the pendulum.
        cart_boundary (float): The boundary for the cart's position
        g (float): The acceleration due to gravity (default is 9.81 m/s^2).
        k (float): The damping factor (default is 0.0, no damping).
        T (float) : Fixed duration of each step (default is 0.1s)
        resolution (int): Number of possible actions to take.
        epsilon (float): The exploration rate for the ε-greedy policy.
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        """
        self.max_force = max_force
        self.time_limit = time_limit
        self.m = m
        self.M = M
        self.l = l
        self.cart_boundary = cart_boundary
        self.g = g
        self.k = k
        self.T = T

        # Calculate the number of discretization bins for each state component
        self.num_actions = resolution
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        # Discretize the actions
        self.actions = np.linspace(-self.max_force, self.max_force, self.num_actions)

        # Initialize the Q-table
        self.Q = {}
        self.policy = {}

        self.reset()

    def reset(self):
        """
        Reset the environment to a random initial state.
        Randomizing initial state within a small range around the upright position
        Returns:
        Tuple[float, float, float, float]: The initial state of the system.
        """
        self.state = (
            np.random.uniform(-0.05, 0.05),  # Small random position around the center
            0,  # Starting with zero velocity
            np.pi
            + np.random.uniform(
                -0.05, 0.05
            ),  # Slightly off from the perfectly upright position
            0,  # Starting with zero angular velocity
        )
        self.state = self.discretize_state(self.state)
        self.time_step = 0
        return self.state

    def step(
        self, action: float
    ) -> Tuple[Tuple[float, float, float, float], float, bool]:
        """
        Take a step in the environment using the given action.

        Args:
        action (float): The force applied to the cart.

        Returns:
        Tuple containing the new state, reward, and a boolean indicating if the episode has ended.
        """
        new_state = update_state(
            self.state, action, self.T, self.m, self.M, self.l, self.g, self.k
        )
        reward = self.calculate_reward(new_state)
        done = self.check_termination(new_state) or self.time_step >= self.time_limit
        self.state = new_state
        self.time_step += 1
        return self.discretize_state(new_state), reward, done

    def calculate_reward(self, state: Tuple[float, float, float, float]) -> float:
        """
        Calculate the reward based on the current state.

        Args:
        state (Tuple[float, float, float, float]): The current state of the system.

        Returns:
        float: The calculated reward.
        """
        x, _, theta, _ = state
        error = abs(theta - np.pi)

        reward = -np.exp(error)  # Penalize based on angle error

        # Penalize for going out of bounds or pendulum falling
        if abs(x) > self.cart_boundary or abs(theta - np.pi) > np.pi / 2:
            reward -= 100
        else:
            reward += 1  # Small positive reward for staying within bounds and upright

        return reward

    def check_termination(self, state: Tuple[float, float, float, float]) -> bool:
        """
        Check if the episode should terminate based on the current state.

        Args:
        state (Tuple[float, float, float, float]): The current state of the system.

        Returns:
        bool: True if the episode should terminate, False otherwise.
        """
        x, _, theta, _ = state
        return (
            abs(x) > self.cart_boundary or abs(theta - np.pi) > np.pi / 2
        )  # Terminate if cart is out of bounds or pendulum falls

    def discretize_state(
        self, state: Tuple[float, float, float, float]
    ) -> Tuple[int, int, int, int]:
        """
        Discretize the continuous state into bins.

        Args:
        state (Tuple[float, float, float, float]): The continuous state of the system.

        Returns:
        Tuple[int, int, int, int]: The discretized state.
        """
        x, x_dot, theta, theta_dot = state
        round_x = round(x, 2)
        round_x_dot = round(x_dot, 1)
        round_theta = round(theta)
        round_theta_dot = round(theta_dot, 1)

        return (round_x, round_x_dot, round_theta, round_theta_dot)

    def epsilon_greedy_policy(self, state: Tuple[float, float, float, float]) -> float:
        """
        Choose an action based on the ε-greedy policy.

        Args:
        state (Tuple[float, float, float, float]): The current state of the system.

        Returns:
        float: The action chosen by the policy.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            max_action = max(
                self.actions, key=lambda action: self.Q.get((state, action), 0.0)
            )
            return max_action

    def update_q_values(
        self,
        state: Tuple[float, float, float, float],
        action: float,
        reward: float,
        next_state: Tuple[float, float, float, float],
        next_action: float,
    ) -> None:
        """
        Update the Q-table using the SARSA update rule based on the experience.

        Args:
        state (Tuple[float, float, float, float]): The current state.
        action (float): The action taken in the current state.
        reward (float): The reward received after taking the action.
        next_state (Tuple[float, float, float, float]): The next state.
        next_action (float): The next action chosen by the policy.

        """
        current_q_value = self.Q.get((state, action), 0.0)
        next_q_value = self.Q.get((next_state, next_action), 0.0)

        # Update the Q-value using the SARSA update rule
        updated_q_value = current_q_value + self.alpha * (
            reward + self.gamma * next_q_value - current_q_value
        )

        # Update the Q-table with the new Q-value
        self.Q[(state, action)] = updated_q_value

        best_action = max(self.actions, key=lambda a: self.Q.get((state, a), 0))
        self.policy[state] = best_action


def train_sarsa(env: InvertedPendulumEnv, num_episodes: int):
    """
    Train the agent using the SARSA algorithm.

    Args:
    env (InvertedPendulumEnv): The environment instance.
    num_episodes (int): The number of episodes to train for.
    """
    for episode in tqdm(range(num_episodes), desc="Training"):
        state = env.reset()
        action = env.epsilon_greedy_policy(state)
        done = False

        while not done:
            next_state, reward, done = env.step(action)
            next_action = env.epsilon_greedy_policy(next_state)

            env.update_q_values(state, action, reward, next_state, next_action)

            state, action = next_state, next_action


def animate_system(env: InvertedPendulumEnv, policy):
    """
    Animate the cart and pendulum system for a given number of steps.

    Args:
    env (InvertedPendulumEnv): The environment of the inverted pendulum.
    """

    fig, ax = plt.subplots()
    ax.set_xlim(-env.cart_boundary - 1, env.cart_boundary + 1)
    ax.set_ylim(-2, 2)

    ax.invert_yaxis()

    cart_width = 0.4
    cart_height = 0.2
    cart_y = 0.5
    cart = Rectangle((0, cart_y), cart_width, cart_height, color="blue")
    ax.add_patch(cart)
    (pendulum,) = ax.plot([], [], "r-", linewidth=2)

    left_boundary = ax.axvline(-env.cart_boundary, color="green", linestyle="--")
    right_boundary = ax.axvline(env.cart_boundary, color="green", linestyle="--")

    info_text = ax.text(-env.cart_boundary, 1.8, "", fontsize=9)

    info_text = ax.text(
        -env.cart_boundary - 0.9, 1.8, "", fontsize=9, verticalalignment="bottom"
    )

    def init():
        cart.set_xy((-cart_width / 2, cart_y - cart_height / 2))
        pendulum.set_data([], [])
        info_text.set_text("")
        return cart, pendulum, left_boundary, right_boundary, info_text

    def update(frame):
        if env.time_step >= env.time_limit or env.check_termination(env.state):
            # If episode has ended or termination condition is met, stop animation
            ani.event_source.stop()
            return cart, pendulum, left_boundary, right_boundary, info_text
        action = policy.get(env.state, np.random.choice(env.actions))
        new_state, _, _ = env.step(
            action
        )  # Step the environment with the chosen action
        x, x_dot, theta, theta_dot = new_state

        # Update cart and pendulum positions
        cart_x = x - cart_width / 2
        pendulum_x = [x, x + env.l * np.sin(theta)]
        pendulum_y = [cart_y, cart_y + env.l * np.cos(theta)]

        cart.set_xy((cart_x, cart_y - cart_height / 2))
        pendulum.set_data(pendulum_x, pendulum_y)

        # Update information text
        info = f"Cart Position: {x:.2f}\nCart Velocity: {x_dot:.2f}\nPendulum Angle: {theta:.2f}\nPendulum Angular Velocity: {theta_dot:.2f}\nAction(Force): {action:.2f}"
        info_text.set_text(info)

        return cart, pendulum, left_boundary, right_boundary, info_text

    ani = FuncAnimation(fig, update, frames=env.time_limit, init_func=init, blit=True)
    plt.show()


env = InvertedPendulumEnv(
    max_force=10, time_limit=200, m=1, M=5, l=2, epsilon=0.3, resolution=25
)
train_sarsa(env, num_episodes=7000)
optimal_policy = env.policy
env.reset()
animate_system(env, policy=optimal_policy)

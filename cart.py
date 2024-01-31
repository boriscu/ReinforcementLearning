import random
import math
import matplotlib.pyplot as plt
from tqdm import trange

g = 9.81
k = 1
ANGLE_BOUNDARY = 12
POS_BOUNDARY = 0.25
F = [-0.15, 0, 0.15]  # <- najbolje za ove vrednosti
# F = [-0.3, -0.2, -0.1,  0, 0.1, 0.2, 0.3]
# F = [-1, -0.5, 0, 0.5, 1]
# F = [-1, 0, 1]
TIME = 30
ITERATIONS = 200000
TEST_ITERATION = 5000


class Pole:

    def __init__(self, length: float, mass: float):
        """
        Class Pole represents pole to be balanced.

        param length: length of pole
        param mass: mass of pole
        """
        self.length = length
        self.mass = mass

    def get_length(self):
        """
        Returns length of pole.

        :return: length of pole
        """
        return self.length

    def get_mass(self):
        """
        Returns mass of pole.

        :return: mass of pole
        """
        return self.mass


class Cart:

    def __init__(self, mass: float):
        """
        Represents cart on which pole will be balanced.

        param mass: mass of cart
        """
        self.mass = mass

    def get_mass(self):
        """
        Returns mass of cart

        :return: mass
        """
        return self.mass


class System:

    def __init__(self, pole: Pole, cart: Cart, initial_angle: float, initial_position: float):
        """
        Represents system that contains pole and cart, with initial position and angle given.

        param pole: pole object
        param cart: cart object
        param initial_angle: initial angle of pole
        param initial_position: initial position of cart
        """
        self.pole = pole
        self.cart = cart
        self.angle = initial_angle
        self.position = initial_position

    def get_pole_length(self):
        """
        Returns length of system pole.

        :return: pole length
        """
        return self.pole.get_length()

    def get_pole_mass(self):
        """
        Returns mass of system pole.

        :return: pole mass
        """
        return self.pole.get_mass()

    def get_cart_mass(self):
        """
        Returns mass of system cart.

        :return: cart mass
        """
        return self.cart.get_mass()

    def get_angle(self):
        """
        Returns current angle of system pole.

        :return: pole angle
        """
        return self.angle

    def get_position(self):
        """
        Returns current position of system cart.

        :return: cart position
        """
        return self.position

    def set_angle(self, angle):
        """
        Sets current pole angle on given value.

        :param angle: angle
        """
        self.angle = angle

    def set_position(self, position):
        """
        Sets current cart position on given value.

        :param position: position
        """
        self.position = position


class Environment:

    def __init__(self, system: System, period: float):
        """
        Environment represents mathematical model of system with its state-space variables,
        sampling period, time and cart-pole system.
        Time is updated every time variables are updated (whenever new force is applied).
        Variables are rounded (discretized) after every update of variables.

        param system: cart-pole system object
        param period: sampling period
        """
        self.time = 0
        self.system = system
        self.period = period
        self.variables = self._initialize_variables()
        self._round_values()

    def _initialize_variables(self):
        """
        Initializes state-spaces variables of model based on given on system cart and pole.

        :return: list[state_space_variables] in the next order: position, velocity, angle, angle velocity
        """
        if self.system.get_angle() > ANGLE_BOUNDARY or self.system.get_angle() < -1 * ANGLE_BOUNDARY:
            raise Exception("Invalid initial angle given.")
        else:
            x1 = self.system.get_position()
            x2 = 0
            x3 = self.system.get_angle()
            x4 = 0
        return [x1, x2, x3, x4]

    def update_variables(self, f: float):
        """
        Updates model state-space variables based on model discretization using Euler 1.
        When variables are updated, one period has passed,so it's added to time.

        param f: force to be applied on system
        """
        self.variables[0] += self.period * self.variables[1]  # x1 = x1 + T * x2
        self.variables[1] += self.period * self._function(f)  # x2 = x2 + T * Fx(f)
        self.variables[2] += self.period * self.variables[3]  # x3 = x3 + T * x4
        self.variables[3] += self.period * self._function(f, position=False)  # x4 = x4 + T * Fo(f)
        self.time += self.period
        self._round_values()

    def _function(self, f, position=True):
        """
        Function for calculating velocity / angle velocity update.
        Calculates velocity is parameter position is True and angle velocity if it's False.

        param f: force
        param position: parameter that indicates which variable is calculated.
        """
        m = self.system.get_pole_mass()
        l = self.system.get_pole_length()
        M = self.system.get_cart_mass()
        x3 = self.variables[2]
        x4 = self.variables[3]
        if position:
            a = m * g * math.sin(x3) * math.cos(x3) - (1 + k) * (f + m * l * math.sin(x3) * math.pow(x4, 2))
            b = m * math.pow(math.cos(x3), 2) - (1 + k) * M
            return a / b
        else:
            a = M * g * math.sin(x3) - math.cos(x3) * (f + m * l * math.sin(x3) * math.pow(x4, 2))
            b = (1 + k) * M * l - m * l * math.pow(math.cos(x3), 2)
            return a / b

    def _round_values(self):
        """
        Rounds space-state variables to desired decimals to simulate discrete system.
        """
        self.variables[0] = round(self.variables[0], 2)
        self.variables[1] = round(self.variables[1], 1)
        self.variables[2] = round(self.variables[2])
        self.variables[3] = round(self.variables[3])
        self.time = round(self.time, 1)

    def get_reward(self):
        """
        Returns reward based on current angle of system. If it is outside boundaries
        function return -20 as it should represent failed control.

        :return: reward ( 1 or -20 )
        """
        # print("Current angle:", self.variables[2])
        if (self.variables[2] > ANGLE_BOUNDARY or self.variables[2] < -1 * ANGLE_BOUNDARY) or (
                self.variables[0] > POS_BOUNDARY or self.variables[0] < -1 * POS_BOUNDARY):
            return -20
        else:
            return 1

    def get_time(self):
        """
        Returns time passed since beginning of control.

        :return: time in seconds
        """
        return self.time

    def get_current_variables(self):
        """
        Returns current state-space variables in tuple form.

        :return: tuple(x1,x2,x3,x4)
        """
        current_variables = tuple(val for val in self.variables)
        return current_variables


def one_step(env: Environment, force: float):
    """
    Applies force on system and returns next state variables and reward of that action.

    param env: environment with model of system
    param force: force to be applied on system
    :return: next state state-space variables, reward
    """
    env.update_variables(force)
    return env.get_current_variables(), env.get_reward()


def init_environment():
    """
    Initializes environment with random initial states.

    :return: environment
    """
    pole = Pole(0.2, 0.1)  # 20cm and 100g
    cart = Cart(0.5)  # 0.5kg
    system = System(pole, cart, random.randint(-3, 3), random.uniform(-0.1, 0.1))  # random initial angle and position
    env = Environment(system, period=0.1)  # sampling time between 0.1 and 0.5s with step 0.1

    return env


def get_q_value(q, state, action):
    """
    Returns value of state if it's already in q, number between -3 and -2 if it's not.

    param q: Q estimation dictionary
    param state: state
    param action: action
    :return: value of q[state,action]
    """
    return q.get((state, action), random.uniform(-3, -2))


def random_action():
    """
    Returns random force from possible forces.

    :return: force.
    """
    return random.choice(F)


def greedy_action(q, current_state):
    """
    Returns greedy action for current state.

    param q: Q estimation dictionary
    param current_state: current state
    :return: greedy action
    """
    values = []
    for f in F:
        values.append((f, get_q_value(q, current_state, f)))
    best_action = max(values, key=lambda x: x[1])[0]  # max returns best tuple, first elem is best actiom
    return best_action


def linear_decay_epsilon(iteration):
    """
    Calculates probability of random action with linear decay from 0.25 to 0.05
    based on training iteration.

    param iteration: current training iteration
    :return:
    """
    decay_rate = (0.25 - 0.05) / ITERATIONS
    current_epsilon = 0.25 - decay_rate * iteration
    current_epsilon = max(current_epsilon, 0.05)
    return current_epsilon


def exploratory_action(q, current_state, iteration):
    """
    Returns exploratory action:
        Random action for probability P = exploratory
        Greedy action probability 1-P

    param q: estimated q function
    param p: current state of variables
    param iteration: iteration of learning process
    :return: action (force)
    """
    exploratory = linear_decay_epsilon(iteration)
    if random.random() < exploratory:
        return random_action()
    return greedy_action(q, current_state)


def update(q, current_state, next_state, action, next_action, reward, iteration, gamma=0.9, alpha=0.1):
    """
    Implementation of SARSA algorithm.
    Q(s,a) <- Q(s,a) + alpha * (r + gamma * Q(s+,a+) - Q(s,a))

    param q: Q estimation dictionary
    param current_state: current state of cart-pole system, which value will be updated
    param next_state: next state after applied action
    param action: applied action
    param next_action: next action
    param reward: reward given
    param iteration: iteration of learning, used for gamma only
    param gamma: ponder
    param alpha: learning rate
    :return: updated Q estimation dictionary
    """
    current_state_q_value = get_q_value(q, current_state, action)
    next_state_q_value = get_q_value(q, next_state, next_action)
    # target = reward + math.pow(gamma, iteration) * next_state_q_value
    target = reward + gamma * next_state_q_value
    updated_current_value = current_state_q_value + alpha * (target - current_state_q_value)
    q[current_state, action] = updated_current_value
    return q


def train():
    """
    Training of cart-pole control with SARSA.

    :return: final Q estimation dictionary
    """
    q = dict()

    for iteration in trange(ITERATIONS):
        """
        In every iteration new environment is made and it is balanced until it falls
        or until enough time has passed ( determined by parameter TIME ).
        """

        environment = init_environment()
        current_state = environment.get_current_variables()
        action = exploratory_action(q, current_state, iteration)

        # end - True if enough time has passed for system to be considered validly controlled.
        # fallen - True if pole has fallen out of boundaries.
        end = fallen = False
        i = 0

        while not (fallen or end):

            next_state, reward = one_step(environment, action)

            next_action = exploratory_action(q, current_state, iteration)
            q = update(q, current_state, next_state, action, next_action, reward, i)

            current_state = next_state
            action = next_action

            if reward < 0:
                fallen = True

            if environment.get_time() > TIME:
                end = True
            i += 1

    return q


def test(f_q):
    """
    Testing control of cart-pole system on final estimated Q.
    Prints percentage of good and bad controlled systems.

    param f_q: final estimation of Q
    """

    good = bad = 0

    for iteration in range(TEST_ITERATION):

        environment = init_environment()
        current_state = environment.get_current_variables()
        action = greedy_action(f_q, current_state)
        e = False

        while not e:

            next_state, reward = one_step(environment, action)
            action = greedy_action(f_q, current_state)
            # if action != -0.15:
            #    print(action)

            if reward < 0:
                bad += 1
                e = True

            if environment.get_time() > TIME:
                good += 1
                e = True

    print("\n   Final test results: ")
    print(f"Successful controls {good} ( {round(good / TEST_ITERATION * 100, 2)}% )")
    print(f"Unsuccessful controls {bad} ( {round(bad / TEST_ITERATION * 100, 2)}% )")
    print("     Testing is finished. \n")


final_q = train()
test(final_q)

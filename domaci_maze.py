import math
import random as rdm
from abc import ABC, abstractmethod
from typing import Iterable, Callable
from copy import copy

import numpy as np
import matplotlib.pyplot as plt

from random import random, choices, randint, sample
import networkx as nx

from itertools import chain


class Node(ABC):

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def get_position(self) -> tuple:
        return (self.x, self.y)

    @abstractmethod
    def get_reward(self) -> float:
        pass

    def is_steppable(self) -> bool:
        return True

    def is_terminal(self) -> bool:
        return False

    def has_value(self) -> bool:
        return True


class RegularNode(Node):

    def __init__(self, reward: float, x: int, y: int):
        super().__init__(x, y)
        self.reward = reward

    def get_reward(self) -> float:
        return self.reward


class TerminalNode(Node):

    def __init__(self, reward: float, x: int, y: int):
        super().__init__(x, y)
        self.reward = reward

    def get_reward(self) -> float:
        return self.reward

    def is_terminal(self) -> bool:
        return True

    def has_value(self) -> bool:
        return False


class WallNode(Node):

    def __init__(self, x: int, y: int):
        super().__init__(x, y)

    def get_reward(self) -> float:
        return 0

    def is_steppable(self) -> bool:
        return False

    def has_value(self) -> bool:
        return False


class TeleportNode(Node):

    def __init__(self, reward: float, x: int, y: int):
        super().__init__(x, y)
        self.reward = reward

    def get_reward(self) -> float:
        return self.reward


"""
    Lavirint je implementiran kao graf tj recnik. Svaki kljuc u recniku predstavlja jedan
    cvor grafa, dok su vrednosti kljuceva list u kojima se nalaze manje liste koje sadrza
    naredni cvor i verovatnocu prelaska u taj cvor:

        graph[node] = list[list[next_node, probability]

    Kreiranjem MazeEnviorment objekta se generise nasumican graf kao atribut. Kao argument
    treba proslediti tuple(height, width) za visinu i sirinu lavirinta. Takodje, atribut 
    current_node predstavlja trenutni objekat cvora, koji sadrzi informacije o poziciji.

    Metodom move_from se na osnovu zadatih verovatnoca prelazi u naredno stanje. Povratna 
    vrednost jeste naredni cvor.

    Metoda print_graph ispisuje graf, ali treba naglasiti da ona ne ispisuje tacan graf vec:

        graph[node.get_position()] = list[list[next_node.get_position(), probability]

"""


class MazeEnvironment:

    def __init__(self, dimensions: tuple[int, int], initial_position: tuple[int, int]):
        if (initial_position[0] > dimensions[0] or initial_position[0] < 1) or (
                initial_position[1] > dimensions[1] or initial_position[1] < 1):
            raise Exception(" Invalid initial position given.\n")
        self.graph_height = dimensions[0]
        self.graph_width = dimensions[1]
        self.graph = self.initialize_graph(self.graph_height, self.graph_width)
        self.current_node = self.set_init_position(initial_position)

    def initialize_graph(self, width, height):
        graph = {}
        total_nodes = height * width
        terminal = 0  # flag for terminal node

        for w in range(1, width + 1):
            for h in range(1, height + 1):
                node = self.generate_random_node(w, h)
                total_nodes -= 1
                graph[node] = []

                if isinstance(node, TerminalNode):
                    terminal += 1

        # if there is no terminal node in random graph, it is manually made, but not from wall node
        if not terminal:
            graph_list = list(graph)
            random_node = self.random_not_wall(graph_list)
            terminal_node = TerminalNode(-1, random_node.get_position()[0], random_node.get_position()[1])
            graph.pop(random_node)
            graph[terminal_node] = []

        for node in graph:
            graph[node] = self.set_probabilities(node, graph)

        return graph

    def set_probabilities(self, node, g):
        probabilities = []

        total_cells = self.graph_width * self.graph_height
        zero_cells = total_cells // 2  # number of cells with zero prob of stepping

        if isinstance(node, WallNode) or isinstance(node, TerminalNode):
            return []

        nodes_list = [node for node in g]

        for n in nodes_list:
            if isinstance(n, WallNode):
                probabilities.append([n, 0])
                zero_cells -= 1
                nodes_list.remove(n)

        if isinstance(node, RegularNode):  # for teleport nodes there is no zero probabilities, except for Walls
            while zero_cells != 0:
                random_node = rdm.choice(nodes_list)
                nodes_list.remove(random_node)
                probabilities.append([random_node, 0])
                zero_cells -= 1

        non_zero_probabilities = self.generate_probabilities(nodes_list)

        for i in range(len(nodes_list)):
            probabilities.append([nodes_list[i], non_zero_probabilities[i]])

        return probabilities

    @staticmethod
    def generate_random_node(w, h):
        prob = rdm.randint(1, 18)
        if prob < 11:
            return RegularNode(-1, w, h)
        elif prob < 13:
            return RegularNode(-10, w, h)
        elif prob < 15:
            return TerminalNode(-1, w, h)
        elif prob < 17:
            return WallNode(w, h)
        else:
            return TeleportNode(-2, w, h)

    @staticmethod
    def generate_probabilities(cells):
        probabilities = np.random.rand(len(cells))
        probabilities /= sum(probabilities)
        return probabilities

    def print_graph(self):
        values_graph = {}
        for node in self.graph:
            values_graph[node.get_position()] = []
            for [next_node, prob] in self.graph[node]:
                values_graph[node.get_position()].append([next_node.get_position(), prob])
        self.print_values(values_graph)
        return

    def print_values(self, g):
        print('\n   ------------------  MAZE GRAPH --------------  ')
        print(' ')
        for node in g:
            if self.get_current_pos_node(node).get_reward() == -10:
                print(node, "* : ", g[node])
                print(' ')
            else:
                print(node, ": ", g[node])
                print(' ')
        print('   ----------------------------------------------  ')
        return

    # returns next node and reward
    def move_from(self, node: Node):
        next_node_index = self.chose_next_node(node)
        self.current_node = self.graph[node][next_node_index][0]
        return self.current_node, self.current_node.get_reward()

    def chose_next_node(self, node: Node):
        probabilities = [pair[1] for pair in self.graph[node]]
        index = np.random.choice(len(probabilities), p=probabilities)
        return index

    def is_terminal(self, node: Node):
        return node.is_terminal()

    def is_terminal_pos(self, position):
        for g in self.graph:
            if g.get_position() == position:
                return g.is_terminal()

    def get_graph(self):
        return self.graph

    def get_current_node(self):
        return self.current_node

    def get_current_pos_node(self, pos):
        for g in self.graph:
            if g.get_position() == pos:
                return g
        raise Exception("Invalid position given.")

    def set_init_position(self, position: tuple[int, int]):
        for node in self.graph:
            if node.get_position() == position:
                return node

    # recursive search for node that is not wall cell
    def random_not_wall(self, g):
        random_node = rdm.choice(g)
        if isinstance(random_node, WallNode):
            random_node = self.random_not_wall(g)
        return random_node

    # returns list[tuple] of probabilities (that are not zero) for node
    def get_actions_probs(self, node: Node):
        return [(pair[0], pair[1]) for pair in self.graph[node] if pair[1] != 0]


def find_terminal(e: MazeEnvironment):
    iteration = 1
    gain = 0
    gamma = 0.9
    current_node = e.get_current_node()
    initial_node_p = e.get_actions_probs(current_node)
    print(initial_node_p)
    print('\n')
    if e.is_terminal(current_node):
        print(f"\nFound terminal node at position {e.get_current_position()} in iteration {iteration} \n")
        print(f"Final gain is: {gain}")
        plot_maze_graph(e, current_node.get_position())
        return
    print(f"Current position, iteration and gain: {e.get_current_position()} , {iteration}, {gain}")
    plot_maze_graph(e, current_node.get_position())
    while True:
        current_node, reward = e.move_from(current_node)
        iteration += 1
        gain += reward * math.pow(gamma, iteration - 1)
        if e.is_terminal(current_node):
            print(f"\nFound terminal node at position {e.get_current_position()} in iteration {iteration}.")
            print(f"Final gain is: {gain}.")
            plot_maze_graph(e, current_node.get_position())
            return
        print(f"Current position, iteration and gain: {e.get_current_position()} , {iteration}, {gain}")
        plot_maze_graph(e, current_node.get_position())


def get_node_color(cell):
    if isinstance(cell, RegularNode) and cell.get_reward() == -10:
        return "red"
    elif isinstance(cell, RegularNode) and cell.get_reward() == -1:
        return "gray"
    elif isinstance(cell, WallNode):
        return "black"
    elif isinstance(cell, TerminalNode):
        return "blue"
    else:
        return "green"


def plot_maze_graph(env: MazeEnvironment, current_position: tuple[int, int]):
    g = nx.DiGraph()
    graph = env.get_graph()
    current_node = None

    for node in graph:
        position = node.get_position()
        if node not in g:
            g.add_node(node, pos=position)

    node_colors = {node: get_node_color(node) for node in g.nodes()}

    plt.figure(figsize=(10, 7))

    nx.draw_networkx_nodes(
        g,
        pos=nx.get_node_attributes(g, 'pos'),
        node_color=[node_colors[n] for n in g.nodes()],
        node_size=700,
    )

    for node in graph:
        if node.get_position() == current_position:
            current_node = node

    nx.draw_networkx_labels(
        g,
        pos=nx.get_node_attributes(g, 'pos'),
        labels={current_node: 'X'},
        font_size=10,
        font_color='white',
        font_weight='bold',
        verticalalignment='center',
        horizontalalignment='center',
    )

    plt.show()


def init_values(env: MazeEnvironment):
    return {s.get_position(): -20 * random() if not env.is_terminal(s) else 0 for s in env.get_graph()}


def update_state_value(env: MazeEnvironment, position, values, gamma):
    # print(" -------------------------------------------- ")
    # print(f"\nCurrent values for position {position}:\n ", values)
    possible_values = []
    current_node = env.get_current_pos_node(position)
    # print("Current node position: ", current_node.get_position())
    p = env.get_actions_probs(current_node)
    # print("\nVerovatnoce prelazaka: ", p)
    for next_node, prob in p:  # nikada ne udje ovde ili lose uradi?
        x = prob * (next_node.get_reward() + gamma * values[next_node.get_position()])
        possible_values.append(x)
        # print(f"{next_node.get_position()} i verovatnoca {prob}")
    # print(possible_values, "\n")
    return sum(possible_values) if possible_values else -1


def async_update_all_values(env: MazeEnvironment, values, gamma):
    # print("\nValues inside 1:\n ", values)
    for s in values:
        if not env.is_terminal_pos(s):
            values[s] = update_state_value(env, s, values, gamma)
    return copy(values)


def value_iteration(env, gamma, eps, v0=None, maxit=100):
    values = v0 if v0 is not None else init_values(env)
    for k in range(maxit):
        print(f"\nIteration: {k + 1}")
        print("Old values: ", values)
        values_copy = copy(values)
        new_values = async_update_all_values(env, values, gamma)
        print("New values: ", new_values)
        err = max([abs(new_values[s] - values_copy[s]) for s in values])
        print("Error is: ", err)
        if err < eps:
            return new_values, k + 1
        values = new_values
    return values, k + 1


initial = (1, 1)
dims = (2, 3)

en = MazeEnvironment(dims, initial)
en.print_graph()

v, ite = value_iteration(en, 1, 0.01)
print(f"\nFinished value iteration on iteration: {ite}")
print(v)

'''
def update_state_value(env: MazeEnvironment, s, v, gamma):
    rhs = []
    cell = env.board[s[0], s[1]]
    for a in env.get_possible_actions(cell, *s):
        s_new, r, _ = env(s, a)
        rhs.append(r + gamma * v.get(s_new, 0))  # Default to 0 if s_new not in v
    return max(rhs) if rhs else 0  # Handle case with no actions


def async_update_all_values(env: MazeEnvironment, v, gamma):
    """Update values of all states.

    Args:
        env (MazeEnvironment): The environment to work on.
        v : Values of other states.
        gamma : discount factor.
    """
    for s in env.get_states():
        if not env.is_terminal(s):
            v[s] = update_state_value(env, s, v, gamma)
    return copy(v)


def init_values(env):
    """Randomly initialize states of the given environment."""
    values = {s: -10 * random() for s in env.get_states()}

    for s in values:
        if env.is_terminal(s):
            values[s] = 0

    return values


def draw_values(env, values, ax=None):
    ax = ax if ax is not None else plt
    draw_board(env.board, ax=ax)
    for s in values:
        ax.text(s[1] - 0.25, s[0] + 0.1, f"{values[s]:.1f}")


values = init_values(env)
draw_values(env, values)

nrows, ncols = 2, 2
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
axes = axes.flatten()
values = init_values(env)
for k, ax in enumerate(axes):
    draw_values(env, values, ax=ax)
    ax.set_title(f"it={k}")
    values = async_update_all_values(env, values, 1.0)


def value_iteration(env, gamma, eps, v0=None, maxiter=100):
    v = v0 if v0 is not None else init_values(env)
    for k in range(maxiter):
        nv = async_update_all_values(env, values, gamma)
        err = max([abs(nv[s] - v[s]) for s in v])
        if err < eps:
            return nv, k
        v = nv
    return v, k


fin_v, k = value_iteration(env, 1.0, 0.01)
draw_values(env, fin_v)
plt.title(f"Converged after {k} iterations")


def greedy_action(env, s, v, gamma):
    action_values = []
    cell = env.board[s[0], s[1]]
    for a in env.get_possible_actions(cell, *s):
        s_next, r, _ = env(s, a)
        action_values.append((a, r + gamma * v.get(s_next, 0)))

    if action_values:
        return max(action_values, key=lambda x: x[1])[
            0
        ]  # Return the action of the max value pair
    else:
        return None  # No action available


def optimal_policy(env, v, gamma):
    return {
        s: greedy_action(env, s, v, gamma)
        for s in env.get_states()
        if not env.is_terminal(s)
    }


aopt = greedy_action(env, (5, 4), fin_v, 1.0)


def action_symbol(a):
    if a == "right":
        return "→"
    elif a == "up":
        return "↑"
    elif a == "left":
        return "←"
    elif a == "down":
        return "↓"
    else:
        raise "Unknown action"


def draw_policy(env, policy, ax=None):
    ax = ax if ax is not None else plt
    draw_board(env.board, ax=ax)
    for s, a in policy.items():
        if s in env.get_states() and a in env.get_possible_actions(
            env.board[s[0], s[1]], *s
        ):
            ax.text(s[1] - 0.25, s[0] + 0.1, action_symbol(a), fontsize=12)


def main():
    # Initialize your environment and graph
    env = MazeEnvironment()
    G = create_graph(env)

    # Start position
    current_position = (0, 0)

    while True:
        # Plot the graph with the current position
        plot_maze_graph(G, env, current_position)

        # Check if the current position is terminal
        if env.is_terminal(current_position):
            print("Reached a terminal cell. Exiting the game.")
            break

        # Display available actions
        display_available_actions(env, *current_position)

        # Get user input for the action
        action = (
            input("Enter action (LEFT, RIGHT, UP, DOWN, or 'exit' to quit): ")
            .strip()
            .lower()
        )

        # Check for exit command
        if action == "exit":
            print("Exiting the game.")
            break

        try:
            # Try to move
            new_position = env.move_from(*current_position, action)
            current_position = new_position  # Update position if move is successful
        except Exception as e:
            # Handle invalid actions
            print("Invalid acction, please try again")
            continue  # Continue the loop, asking for input again

'''
# Simulate the maze
# main()

# Test finding optimal polici
# pi = optimal_policy(env, fin_v, 1.0)
# draw_policy(env, pi)
# plt.show()

# Zadatak level 3:

# Assume that the Maze is graph-like and not grid-like, so that there are different possible actions to take from each cell.
# Add TELEPORT CELLS and modify the code to accomodate this kind of cells also.
# Implement value iteration algorithm using Q function instead of V function.
# Implement policy iteration algorithm using both V and Q functions.
# In all cases update, modify, and add visualization facilities to illustrate correctness of the implementation.

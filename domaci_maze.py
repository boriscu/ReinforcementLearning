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
    """Abstract base class for all maze cells."""

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def get_position(self) -> tuple:
        return (self.x, self.y)

    @abstractmethod
    def get_reward(self) -> float:
        """The reward an agent receives when stepping onto this cell."""
        pass

    def is_steppable(self) -> bool:
        """Checks if an agent can step onto this cell.

        Regular and terminal cells are steppable.
        Walls are not steppable.
        """
        return True

    def is_terminal(self) -> bool:
        """Checks if the cell is terminal.

        When stepping onto a terminal cell the agent exits
        the maze and finishes the game.
        """
        return False

    def has_value(self) -> bool:
        """Check if the cell has value.

        The value is defined for regular cells and terminal cells,
        but not for walls.
        """
        return True


class RegularNode(Node):
    """A common, non-terminal, steppable cell."""

    def __init__(self, reward: float, x: int, y: int):
        super().__init__(x, y)
        self.reward = reward

    def get_reward(self) -> float:
        return self.reward


class TerminalNode(Node):
    """A terminal cell."""

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
    """A non-steppable cell."""

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
def default_cell_color(: ) -> tuple[int, int, int]:
    if isinstance(cell, RegularCell):
        if cell.get_reward() == -1:
            return (255, 255, 255)            # Regular cell
        else:
            return (255, 0, 0)                # Regular cell with penalty
    elif isinstance(cell, WallCell):
        return (0, 0, 0)                      # Wall cell
    elif isinstance(cell, TerminalCell):
        return (0, 0, 255)                    # Terminal cell
    else:
        return (0, 255, 0)                    # Teleport Cell
"""

"""
    Lavirint je implementiran kao graf tj recnik. Svaki kljuc u recniku predstavlja jedan
    cvor grafa, dok su vrednosti kljuceva list u kojima se nalaze manje liste koje sadrza
    naredni cvor i verovatnocu prelaska u taj cvor:

        graph[node] = list[list[next_node, probability]

    Kreiranjem MazeEnviorment objekta se generise nasumican graf kao atribut. Kao argument
    treba proslediti tuple(height, width) za visinu i sirinu lavirinta.
"""


class MazeEnvironment:
    """Wrapper for a maze board that behaves like an MDP environment.

    This is a callable object that behaves like a deterministic MDP state
    transition function: given the current state and action, it returns the
    following state and reward.

    In addition, the environment object is capable of enumerating all possible
    states and all possible actions. For a given state it is also capable of
    deciding if the state is terminal or not.
    """

    def __init__(self, dimensions: tuple[int, int]):
        """Initialize the enviornment by specifying the underlying maze board."""
        self.graph_height = dimensions[0]
        self.graph_width = dimensions[1]
        self.graph = self.initialize_graph(self.graph_height, self.graph_width)

    def initialize_graph(self, width, height):
        graph = {}
        total_nodes = height * width
        terminal = 0  # flag for terminal node, if there is no terminal node in random graph, it is manually made

        for w in range(1, width + 1):
            for h in range(1, height + 1):
                node = self.generate_random_node(w, h)
                total_nodes -= 1
                graph[node] = []

                if isinstance(node, TerminalNode):
                    terminal += 1

        if not terminal:
            graph_list = list(graph)
            random_node = rdm.choice(graph_list)
            terminal_node = TerminalNode(0, random_node.get_position()[0], random_node.get_position()[1])
            graph.pop(random_node)
            graph[terminal_node] = []

        for node in graph:
            graph[node] = self.set_actions_probability(node, graph)

        return graph

    def set_actions_probability(self, node, g):
        actions_probabilities = []

        total_cells = self.graph_width * self.graph_height
        zero_cells = total_cells // 2  # number of cells with zero prob of stepping

        if isinstance(node, WallNode) or isinstance(node, TerminalNode):
            return []

        nodes_list = [node for node in g]

        for n in nodes_list:
            if isinstance(n, WallNode):
                actions_probabilities.append([n, 0])
                zero_cells -= 1
                nodes_list.remove(n)

        if isinstance(node, RegularNode):  # for teleport nodes there is no zero probabilities, except for Walls
            while zero_cells != 0:
                random_node = rdm.choice(nodes_list)
                nodes_list.remove(random_node)
                actions_probabilities.append([random_node, 0])
                zero_cells -= 1

        non_zero_probabilities = self.generate_probabilities(nodes_list)

        for i in range(len(nodes_list)):
            actions_probabilities.append([nodes_list[i], non_zero_probabilities[i]])

        return actions_probabilities

    @staticmethod
    def generate_random_node(w, h):
        prob = rdm.randint(1, 18)
        if prob < 11:
            return RegularNode(-1, w, h)
        elif prob < 13:
            return RegularNode(-10, w, h)
        elif prob < 15:
            return TerminalNode(0, w, h)
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

    @staticmethod
    def print_values(g):
        print('\n   ------------------  MAZE GRAPH --------------  ')
        print(' ')
        for node in g:
            print(node, ": ", g[node])
            print(' ')
        print('   ----------------------------------------------  ')
        return

    '''
    def validate_position(self, row, col):
        """A utility function that validates a position."""
        if row < 0 or row >= self.rows_no:
            raise Exception("Invalid row position.")
        if col < 0 or col >= self.cols_no:
            raise Exception("Invalid column position.")
        if not self.board[row, col].is_steppable():
            raise Exception("Invalid position: unsteppable cell.")
        return row, col
    '''

    # returns next node
    def move_from(self, node: Node):
        next_node_index = self.chose_next_node(node)
        return self.graph[node][next_node_index][0]

    def chose_next_node(self, node: Node):
        probabilities = [pair[1] for pair in self.graph[node]]
        print(probabilities)
        index = np.random.choice(len(probabilities), p=probabilities)
        return index

    '''
    def __call__(
        self, state: tuple[int, int], action: str
    ) -> tuple[tuple[int, int], float, bool]:
        row, col = state
        new_row, new_col = self.move_from(row, col, action)
        new_cell = self.board[new_row, new_col]
        reward = new_cell.get_reward()
        is_terminal = new_cell.is_terminal()
        return (new_row, new_col), reward, is_terminal

    def get_states(self):
        states = []
        for r in range(self.board.rows_no):
            for c in range(self.board.cols_no):
                if self.board[r, c].is_steppable():
                    states.append((r, c))
        return states

    def is_terminal(self, s):
        return self.board[s[0], s[1]].is_terminal()

    def get_actions(self, cell):
        if cell in self.graph:
            return list(self.graph[cell].keys())
        else:
            return []
    '''

'''
def create_graph(maze_env):
    G = nx.DiGraph()
    for cell, actions in maze_env.graph.items():
        for action, target in actions.items():
            source = (
                cell.row,
                cell.col,
            )  # assuming each cell has row and col attributes
            G.add_edge(source, target, action=action)
    return G
'''

dims = (3, 3)
env = MazeEnvironment((dims))
env.print_graph()

'''
def get_node_color(cell):
    if isinstance(cell, RegularCell) and cell.get_reward() == -10:
        return "red"
    elif isinstance(cell, RegularCell) and cell.get_reward() == -1:
        return "gray"
    elif isinstance(cell, WallCell):
        return "black"
    elif isinstance(cell, TerminalCell):
        return "blue"
    # Add more as needed


def plot_maze_graph(G, maze_env, current_position):
    pos = {
        (r, c): (c, -r)
        for r in range(maze_env.board.rows_no)
        for c in range(maze_env.board.cols_no)
    }

    # Add all cells to the graph as nodes
    for r in range(maze_env.board.rows_no):
        for c in range(maze_env.board.cols_no):
            cell = maze_env.board[r, c]
            if (r, c) not in G:
                G.add_node(
                    (r, c)
                )  # Add cell as a node if it's not already in the graph

    # Create a dictionary for node colors
    node_colors = {(r, c): get_node_color(maze_env.board[r, c]) for r, c in G.nodes()}

    plt.figure(figsize=(12, 8))

    # Separate nodes based on whether they have actions or not
    action_nodes = set(u for u, v, d in G.edges(data=True))
    no_action_nodes = set(G.nodes()) - action_nodes

    # Draw nodes with actions (as circles)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=action_nodes,
        node_color=[node_colors[n] for n in action_nodes],
        node_size=700,
    )

    # Draw nodes without actions (as squares)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=no_action_nodes,
        node_color=[node_colors[n] for n in no_action_nodes],
        node_size=700,
        node_shape="s",
    )

    # Drawing edges
    nx.draw_networkx_edges(G, pos, edge_color="black", arrows=False)

    current_pos = pos[current_position]
    plt.text(
        current_pos[0],
        current_pos[1],
        "X",
        color="black",
        fontsize=12,
        ha="center",
        va="center",
    )

    plt.show()


def display_available_actions(maze_env, row, col):
    cell = maze_env.board[row, col]
    actions = maze_env.get_possible_actions(cell, row, col)
    print("Available actions:", ", ".join(actions.keys()))


# Commented out Rapajas code for logic
# TODO Make it work with graphs
'''

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
'''

'''
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

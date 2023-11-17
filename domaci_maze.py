from abc import ABC, abstractmethod
from typing import Iterable, Callable
from copy import copy

import numpy as np
import matplotlib.pyplot as plt

from random import random, choices, randint, sample
import networkx as nx


class Cell(ABC):
    """Abstract base class for all maze cells."""

    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col

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


class RegularCell(Cell):
    """A common, non-terminal, steppable cell."""

    def __init__(self, reward: float, row: int, col: int):
        super().__init__(row, col)
        self.reward = reward

    def get_reward(self) -> float:
        return self.reward


class TerminalCell(Cell):
    """A terminal cell."""

    def __init__(self, reward: float, row: int, col: int):
        super().__init__(row, col)
        self.reward = reward

    def get_reward(self) -> float:
        return self.reward

    def is_terminal(self) -> bool:
        return True

    def has_value(self) -> bool:
        return False


class WallCell(Cell):
    """A non-steppable cell."""

    def __init__(self, row: int, col: int):
        super().__init__(row, col)

    def get_reward(self) -> float:
        return 0

    def is_steppable(self) -> bool:
        return False

    def has_value(self) -> bool:
        return False


class MazeBoard:
    """Rectangular grid of cells representing a single mase."""

    @staticmethod
    def validate_cells(
        cells: Iterable[Iterable[Cell]],
    ) -> tuple[int, int, list[list[Cell]]]:
        """
        Utility function used to validate the given double-iterable of of cells.

        Returns: tuple[int, int, list[list[Cell]]]
            int: number of rows
            int: number of columns
            list[list[Cell]]: double-list of cells from the input iterable
        """
        cells = [list(row) for row in cells] if cells else []
        if not cells:
            raise Exception("Number of rows in a board must be at least one.")
        if not cells[0]:
            raise Exception("There has to be at least one column.")
        rows_no = len(cells)
        cols_no = len(cells[0])
        for row in cells:
            if not row or len(row) != cols_no:
                raise Exception(
                    "Each row in a a board must have the same number of columns. "
                )
        return rows_no, cols_no, cells

    def __init__(self, cells: Iterable[Iterable[Cell]]):
        """
        Initialize the maze board from the given `cells`.

        The input double-iterable of cells is such that the elements of the outer
        iterable are considered to be rows. The input cells must satisfy certain
        conditions in order to be considered valid:

        * There must be at least one row.
        * All rows must be of the same length, which is at least one.
        """
        rows_no, cols_no, cells = MazeBoard.validate_cells(cells)
        self.cells = cells
        self.rows_no = rows_no
        self.cols_no = cols_no

    def __getitem__(self, key: tuple[int, int]) -> Cell:
        """Return cell in the given row and column."""
        r, c = key
        return self.cells[r][c]


# Type-hint used to indicate a function which, when invoked, generates a cell.
CellGenerator = Callable[[], Cell]


def create_random_board(
    size: tuple[int, int], specs=list[tuple[float, CellGenerator]]
) -> MazeBoard:
    h, w = size
    weights = [w for w, _ in specs]
    generators = [g for _, g in specs]

    def random_cell(row, col):  # Updated to include row and col
        cell_generator = choices(generators, weights, k=1)[0]
        return cell_generator(row, col)  # Pass row and col

    cells = [[random_cell(i, j) for j in range(w)] for i in range(h)]
    return MazeBoard(cells)


def default_cell_color(cell: Cell) -> tuple[int, int, int]:
    if isinstance(cell, RegularCell):
        if cell.get_reward() == -1:
            return (255, 255, 255)  # Regular cell
        else:
            return (255, 0, 0)  # Regular cell with penalty
    elif isinstance(cell, WallCell):
        return (0, 0, 0)  # Wall cell
    else:
        return (0, 0, 255)  # Terminal cell


def draw_board(
    board: MazeBoard, color=default_cell_color, pos: tuple[int, int] = None, ax=None
):
    ax = ax if ax is not None else plt
    board_img = np.ones(shape=(board.rows_no, board.cols_no, 3), dtype=np.uint8)
    for i in range(board.rows_no):
        for j in range(board.cols_no):
            board_img[i, j, :] = color(board[i, j])
    if pos is not None:
        row, col = pos
        ax.text(col - 0.1, row + 0.1, "X", fontweight="bold")
    if ax is not None:
        ax.imshow(board_img)
    else:
        plt.imshow(board_img)


# This will be used as the default specification for the generator...
# It requires that the reqular cells with reward -1 are 7 times more likely than
# regular cells with higher penalty, walls and terminal cells.

DEFAULT_SPECS = [
    (10, lambda row, col: RegularCell(-1, row, col)),
    (2, lambda row, col: RegularCell(-10, row, col)),
    (2, lambda row, col: WallCell(row, col)),
    (1, lambda row, col: TerminalCell(-1, row, col)),
]

board = create_random_board(size=(8, 8), specs=DEFAULT_SPECS)

# Let us first enumerate the possible actions for better readability.
RIGHT = 0
UP = 1
LEFT = 2
DOWN = 3

# The following lus is used only to provide human-readable form of each action
ACTIONS = ["RIGHT", "UP", "LEFT", "DOWN"]


class MazeEnvironment:
    """Wrapper for a maze board that behaves like an MDP environment.

    This is a callable object that behaves like a deterministic MDP state
    transition function: given the current state and action, it returns the
    following state and reward.

    In addition, the environment object is capable of enumerating all possible
    states and all possible actions. For a given state it is also capable of
    deciding if the state is terminal or not.
    """

    def __init__(self, board: MazeBoard):
        """Initialize the enviornment by specifying the underlying maze board."""
        self.board = board
        self.graph = self.initialize_graph()

    def initialize_graph(self):
        graph = {}
        for r in range(self.board.rows_no):
            for c in range(self.board.cols_no):
                cell = self.board[r, c]
                graph[cell] = self.get_possible_actions(cell, r, c)
        return graph

    def get_possible_actions(self, cell, row, col):
        potential_actions = {}

        if isinstance(cell, WallCell):
            return {}

        # Check all possible directions and add them if they lead to a steppable cell
        if row > 0 and self.board[row - 1, col].is_steppable():
            potential_actions["up"] = (row - 1, col)
        if row < self.board.rows_no - 1 and self.board[row + 1, col].is_steppable():
            potential_actions["down"] = (row + 1, col)
        if col > 0 and self.board[row, col - 1].is_steppable():
            potential_actions["left"] = (row, col - 1)
        if col < self.board.cols_no - 1 and self.board[row, col + 1].is_steppable():
            potential_actions["right"] = (row, col + 1)

        # # Randomly select a subset of these potential actions
        # num_actions = randint(
        #     1, len(potential_actions)
        # )  # At least 1 action, up to the number of potential actions
        # actions = dict(
        #     sample(
        #         list(potential_actions.items()),
        #         min(num_actions, len(potential_actions)),
        #     )
        # )

        return potential_actions

    def validate_position(self, row, col):
        """A utility function that validates a position."""
        if row < 0 or row >= self.rows_no:
            raise Exception("Invalid row position.")
        if col < 0 or col >= self.cols_no:
            raise Exception("Invalid column position.")
        if not self.board[row, col].is_steppable():
            raise Exception("Invalid position: unsteppable cell.")
        return row, col

    def move_from(self, row: int, col: int, action: str) -> tuple[int, int]:
        if action in self.graph[self.board[row, col]]:
            return self.graph[self.board[row, col]][action]
        else:
            raise Exception(f"Invalid action: {action} for cell ({row}, {col}).")

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

# env = MazeEnvironment(board)

# def update_state_value(env: MazeEnvironment, s, v, gamma):
#     """Update value of the given state.

#     Args:
#         env (MazeEnvironment): The environment to work on.
#         s : The state (cell position).
#         v : Values of other states.
#         gamma : discount factor.
#     """
#     rhs = []
#     for a in env.get_actions():
#         s_new, r, _ = env(s, a)
#         rhs.append(r + gamma * v[s_new])
#     return max(rhs)


# def async_update_all_values(env: MazeEnvironment, v, gamma):
#     """Update values of all states.

#     Args:
#         env (MazeEnvironment): The environment to work on.
#         v : Values of other states.
#         gamma : discount factor.
#     """
#     for s in env.get_states():
#         if not env.is_terminal(s):
#             v[s] = update_state_value(env, s, v, gamma)
#     return copy(v)


# def init_values(env):
#     """Randomly initialize states of the given environment."""
#     values = {s: -10 * random() for s in env.get_states()}

#     for s in values:
#         if env.is_terminal(s):
#             values[s] = 0

#     return values


# def draw_values(env, values, ax=None):
#     ax = ax if ax is not None else plt
#     draw_board(env.board, ax=ax)
#     for s in values:
#         ax.text(s[1] - 0.25, s[0] + 0.1, f"{values[s]:.1f}")

# values = init_values(env)
# draw_values(env, values)

# nrows, ncols = 2, 2
# fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
# axes = axes.flatten()
# values = init_values(env)
# for k, ax in enumerate(axes):
#     draw_values(env, values, ax=ax)
#     ax.set_title(f"it={k}")
#     values = async_update_all_values(env, values, 1.0)


# def value_iteration(env, gamma, eps, v0=None, maxiter=100):
#     v = v0 if v0 is not None else init_values(env)
#     for k in range(maxiter):
#         nv = async_update_all_values(env, values, gamma)
#         err = max([abs(nv[s] - v[s]) for s in v])
#         if err < eps:
#             return nv, k
#         v = nv
#     return v, k


# fin_v, k = value_iteration(env, 1.0, 0.01)
# draw_values(env, fin_v)
# plt.title(f"Converged after {k} iterations")


# def greedy_action(env, s, v, gamma):
#     vs = []
#     for a in env.get_actions():
#         s_next, r, _ = env(s, a)
#         vs.append(r + gamma * v[s_next])
#     return np.argmax(vs)


# aopt = greedy_action(env, (5, 4), fin_v, 1.0)
# ACTIONS[aopt]


# def optimal_policy(env, v, gamma):
#     return {
#         s: greedy_action(env, s, v, gamma)
#         for s in env.get_states()
#         if not env.is_terminal(s)
#     }


# def action_symbol(a):
#     if a == RIGHT:
#         return "→"
#     elif a == UP:
#         return "↑"
#     elif a == LEFT:
#         return "←"
#     elif a == DOWN:
#         return "↓"
#     else:
#         raise "Unknown action"


# def draw_policy(env, policy, ax=None):
#     ax = ax if ax is not None else plt
#     draw_board(env.board, ax=ax)
#     for s, a in policy.items():
#         ax.text(s[1] - 0.25, s[0] + 0.1, action_symbol(a))


def main():
    # Initialize your environment and graph
    env = MazeEnvironment(board)
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


main()


# Zadatak level 3:

# Assume that the Maze is graph-like and not grid-like, so that there are different possible actions to take from each cell.
# Add TELEPORT CELLS and modify the code to accomodate this kind of cells also.
# Implement value iteration algorithm using Q function instead of V function.
# Implement policy iteration algorithm using both V and Q functions.
# In all cases update, modify, and add visualization facilities to illustrate correctness of the implementation.

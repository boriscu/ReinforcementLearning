import random as rdm
from abc import ABC, abstractmethod
from copy import copy

import numpy as np
import matplotlib.pyplot as plt

from random import random, choice

import networkx as nx
from prettytable import PrettyTable


class Node(ABC):
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def get_position(self) -> tuple:
        """
        Gets the position of the node.

        Returns:
            tuple: A tuple (x, y) representing the x and y coordinates of the node.
        """
        return (self.x, self.y)

    @abstractmethod
    def get_reward(self) -> float:
        """
        Gets the reward value associated with the node.

        Returns:
            float: The reward value for the node. Specific value depends on the type of node.
        """
        pass

    def is_steppable(self) -> bool:
        """
        Checks if the node can be stepped on.

        Returns:
            bool: True if the node is steppable, False otherwise.
        """
        return True

    def is_terminal(self) -> bool:
        """
        Checks if the node is a terminal node.

        Returns:
            bool: True if the node is a terminal node, False otherwise.
        """
        return False

    def has_value(self) -> bool:
        """
        Checks if the node holds a value.

        Returns:
            bool: True if the node has a value, False otherwise.
        """
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
    The maze is implemented as a graph, specifically as a dictionary. Each key in the dictionary represents a node in the graph, 
    while the values are lists containing smaller lists. These smaller lists consist of an action, the next node, 
    and the probability of moving to that node when the action is taken:

        graph[node] = list[list[action, next_node, probability]]

    Note: Actions are represented by natural numbers. If an action is 0, 
    it implies that there is no connection between those nodes (hence, the probability will also be 0).

    Creating a MazeEnvironment object generates a random graph as an attribute. 
    The dimensions of the maze (height and width) should be passed as a tuple argument to the constructor.

    The graph is initialized using the initialize_graph function (which is called by the environment's constructor). 
    Within this function, a random graph is formed where probabilities of transitioning to the next nodes are generated for each node in the graph. 
    For terminal nodes and walls, there are no states to transition into from them. For regular nodes, 
    there is a possibility of transitioning to half of the other nodes, while for teleport nodes, 
    there is a possibility to transition to any other node (except walls). 
    More detailed steps are explained inside the function.

    The print_graph method outputs the graph, but it's important to note that it does not display the exact graph structure, 
    but rather a more natural representation:

        graph[node.get_position()] = list[list[action, next_node.get_position(), probability]

    As for the other methods, their functionality is clear from their names.

    Actions are configurable but must be natural numbers. 
    Restrictions on them are written at the point of calling all algorithms.
"""

ACTIONS = [1, 2, 3]


class MazeEnvironment:
    def __init__(self, dimensions: tuple[int, int]):
        """
        Initializes the MazeEnvironment with a specified size.

        Args:
            dimensions (tuple[int, int]): A tuple representing the dimensions (width, height) of the maze.
        """
        self.graph_height = dimensions[0]
        self.graph_width = dimensions[1]
        self.graph = self.initialize_graph(self.graph_height, self.graph_width)

    def initialize_graph(self, width: int, height: int) -> dict:
        """
        Initializes and returns the graph for the maze environment.

        Args:
            width (int): The width of the maze.
            height (int): The height of the maze.

        Returns:
            dict: The graph representing the maze, where keys are node objects and values are lists of connected nodes with probabilities.
        """
        graph = {}
        terminal_node_created = False

        for w in range(1, width + 1):
            for h in range(1, height + 1):
                node = self.generate_random_node(w, h)
                graph[node] = []

                if isinstance(node, TerminalNode):
                    terminal_node_created = True

        # If no terminal node was created, randomly select a non-wall node to convert to a terminal node
        if not terminal_node_created:
            non_wall_nodes = [n for n in graph.keys() if not isinstance(n, WallNode)]
            node_to_replace = choice(non_wall_nodes)
            terminal_node = TerminalNode(-1, node_to_replace.x, node_to_replace.y)
            graph.pop(node_to_replace)
            graph[terminal_node] = []

        # Set probabilities for transitions between nodes
        for node in graph:
            graph[node] = self.set_probabilities(node, graph)

        return graph

    def set_probabilities(self, node: Node, graph: dict) -> list:
        """
        Sets and returns the movement probabilities for the given node to other nodes in the graph.

        Args:
            node (Node): The node for which probabilities are to be set.
            g (dict): The graph of the maze environment.

        Returns:
            list: A list of probabilities for moving from the given node to other nodes.
        """
        # Terminal and wall node do not have any probabilities
        if isinstance(node, (WallNode, TerminalNode)):
            return []

        nodes_list = [n for n in graph if not isinstance(n, WallNode)]
        probabilities = []

        # Calculate the number of non-wall nodes that should have zero probability
        total_cells = self.graph_width * self.graph_height
        zero_cells = min(len(nodes_list), total_cells // 2)

        # Assign zero probability to a subset of non-wall nodes
        for _ in range(zero_cells):
            random_node = rdm.choice(nodes_list)
            nodes_list.remove(random_node)
            probabilities.append([0, random_node, 0])

        # Distribute the remaining nodes among the actions
        for action in ACTIONS:
            if nodes_list:
                action_len = len(nodes_list) // len(ACTIONS) or 1
                nodes_for_action = nodes_list[:action_len]
                nodes_list = nodes_list[action_len:]

                non_zero_probabilities = self.generate_probabilities(nodes_for_action)
                for node, prob in zip(nodes_for_action, non_zero_probabilities):
                    probabilities.append([action, node, prob])

        return probabilities

    @staticmethod
    def generate_random_node(w: int, h: int) -> Node:
        """
        Generates a random node of a specific type based on a probability distribution.

        Args:
            w (int): The x-coordinate of the node.
            h (int): The y-coordinate of the node.

        Returns:
            Node: A randomly generated node of type RegularNode, TerminalNode, WallNode, or TeleportNode.
        """
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
    def generate_probabilities(cells: list) -> np.ndarray:
        """
        Generates and returns a normalized probability distribution for a list of cells.

        This method creates a random probability for each cell and normalizes these probabilities so that they sum up to 1.

        Args:
            cells (list): A list of cells for which probabilities are to be generated.

        Returns:
            np.ndarray: An array of normalized probabilities corresponding to each cell.
        """

        probabilities = np.random.rand(len(cells))
        probabilities /= sum(probabilities)
        return probabilities

    # prints graph (with position)
    def print_graph(self):
        """
        Prints the maze graph in a human-readable format. The graph is represented in terms of positions, actions, and probabilities.
        """
        values_graph = {}
        for node in self.graph:
            values_graph[node.get_position()] = []
            for [action, next_node, prob] in self.graph[node]:
                values_graph[node.get_position()].append(
                    [action, next_node.get_position(), prob]
                )
        self.print_values(values_graph)
        return

    def print_values(self, g: dict):
        """
        Prints the values of the maze graph.

        Args:
            g (dict): The graph of the maze, where keys are node positions and values are lists of actions, next node positions, and probabilities.
        """
        print(
            "\n----------------------------------------------- MAZE GRAPH "
            "------------------------------------------------ "
        )
        print(" ")
        for node in g:
            if self.get_current_pos_node(node).get_reward() == -10:
                print(node, "* : ", g[node])  # for regular node with penalty
                print(" ")
            else:
                print(node, ": ", g[node])
                print(" ")
        return

    # returns True if given node is terminal
    @staticmethod
    def is_terminal(node: Node) -> bool:
        """
        Checks if the given node is a terminal node.

        Args:
            node (Node): The node to be checked.

        Returns:
            bool: True if the node is a terminal node, False otherwise.
        """

        return node.is_terminal()

    # return True if node at position (x,y) is terminal
    def is_terminal_pos(self, position: tuple) -> bool:
        """
        Checks if the node at the given position is a terminal node.

        Args:
            position (tuple): The position of the node.

        Returns:
            bool: True if the node at the given position is terminal, False otherwise.
        """
        for g in self.graph:
            if g.get_position() == position:
                return g.is_terminal()

    # returns graph
    def get_graph(self) -> dict:
        """
        Returns the graph of the maze environment.

        Returns:
            dict: The graph representing the maze.
        """
        return self.graph

    # returns node at position (x,y)
    def get_current_pos_node(self, pos: tuple) -> Node:
        """
        Returns the node at the given position.

        Args:
            pos (tuple): The position of the node.

        Returns:
            Node: The node at the given position.

        Raises:
            Exception: If the position is invalid.
        """
        for g in self.graph:
            if g.get_position() == pos:
                return g
        raise Exception("Invalid position given.")

    # search for node that is not wall node in graph
    def random_not_wall(self, g: list) -> Node:
        """
        Selects a random node from the graph that is not a wall node.

        Args:
            g (list): The list of nodes in the graph.

        Returns:
            Node: A randomly selected node that is not a wall node.
        """
        random_node = rdm.choice(g)
        if isinstance(random_node, WallNode):
            random_node = self.random_not_wall(g)
        return random_node

    def get_action_probabilities(self, node: Node, action: int) -> list:
        """
        Returns a list of tuples containing the next node and the probability of reaching it for a given action from the given node.

        The action is an integer representing a specific kind of move or decision the agent can make from the current node.
        The probabilities are extracted from the graph structure where each node key maps to a list of tuples representing
        possible actions, their corresponding next nodes, and the probabilities of transitioning to those nodes.

        Args:
            node (Node): The node from which the action is taken.
            action (int): The action to be taken.

        Returns:
            list: A list of tuples (next_node, probability) for the given action.
        """
        action_probabilities = []
        for transition in self.graph[node]:
            transition_action, next_node, probability = transition
            if transition_action == action:
                action_probabilities.append((next_node, probability))

        return action_probabilities

    def get_next_node(self, nodes_probs: list) -> Node:
        """
        Selects the next node based on the given probabilities or returns a default node if probabilities are not available.

        Args:
            nodes_probs (list): A list of tuples (node, probability) representing the possible next nodes and their probabilities.

        Returns:
            Node: The selected next node based on the probabilities, or a default node if no valid probabilities are available.
        """
        if not nodes_probs or not any(prob for _, prob in nodes_probs):
            # Return the first node in the list as a default or another predefined node
            return nodes_probs[0][0] if nodes_probs else self.get_default_node()

        probabilities = [probability for _, probability in nodes_probs]
        index = np.random.choice(len(probabilities), p=probabilities)
        return nodes_probs[index][0]

    def get_default_node(self):
        # Return the first node in the graph
        return next(iter(self.get_graph()), None)


def get_node_color(cell: Node) -> str:
    """
    Determines the color representation of a node for visualization.

    Args:
        cell (Node): The node for which the color is to be determined.

    Returns:
        str: The color string representing the node type.
    """
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


def plot_maze_graph(env: MazeEnvironment):
    """
    Plots the maze graph visually using networkx and matplotlib, displaying all nodes, including wall nodes, but excluding edges to or from wall nodes.

    Args:
        env (MazeEnvironment): The maze environment to be plotted.
    """
    g = nx.DiGraph()
    graph = env.get_graph()
    edge_labels = {}
    pos = {}  # Dictionary to hold positions
    offset = 0.03  # Adjust this value to move the label position
    edge_colors = []

    action_colors = {
        1: "magenta",
        2: "orange",
        3: "cyan",
        # Add more colors if needed
    }

    for node in graph:
        position = node.get_position()
        pos[node] = position  # Ensure every node has a position in pos dictionary
        g.add_node(node, pos=position)

        if not isinstance(node, WallNode):  # Skip adding edges for wall nodes
            for action, next_node, probability in graph[node]:
                if (
                    action != 0
                    and probability != 0
                    and not isinstance(next_node, WallNode)
                ):
                    g.add_edge(node, next_node)
                    edge_colors.append(
                        action_colors.get(action, "black")
                    )  # Default to black if action not in mapping
                    edge_labels[(node, next_node)] = f"{next_node.get_reward()}"

    node_colors = {
        node: get_node_color(node) for node in g.nodes()
    }  # Get colors for nodes in g
    node_color_list = [
        node_colors[node] for node in g.nodes()
    ]  # Create color list based on nodes in g

    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(g, pos, node_color=node_color_list, node_size=700)
    nx.draw_networkx_edges(
        g, pos, edge_color=edge_colors, arrowstyle="->", arrowsize=30
    )

    # Adjusting label positions for edge labels
    edge_labels_pos = {
        edge: [
            (pos[edge[0]][0] + pos[edge[1]][0]) / 2 + offset,
            (pos[edge[0]][1] + pos[edge[1]][1]) / 2 + offset,
        ]
        for edge in edge_labels
    }

    for edge, new_pos in edge_labels_pos.items():
        label = edge_labels[edge]
        plt.text(new_pos[0], new_pos[1], label, fontsize=10, color="red")

    # Legends for node types and actions
    node_type_colors = {
        "Regular Node": "gray",
        "Terminal Node": "blue",
        "Wall Node": "black",
        "Teleport Node": "green",
    }
    node_legends = [
        plt.Line2D([0], [0], color=color, marker="o", linestyle="", markersize=10)
        for color in node_type_colors.values()
    ]
    action_legends = [
        plt.Line2D([0], [0], color=color, marker="", linestyle="-")
        for color in action_colors.values()
    ]

    legend1 = plt.legend(
        node_legends, node_type_colors.keys(), title="Node Types", loc="upper left"
    )
    plt.gca().add_artist(legend1)
    plt.legend(
        action_legends,
        [f"Action {a}" for a in action_colors.keys()],
        title="Actions",
        loc="lower left",
    )

    plt.axis("off")
    plt.show()


"""
    When it comes to the implementation of the algorithms, it is done in a standard way, 
    and a more detailed explanation follows:

                                   Value iteration: 

    The estimation of Q-values and V-values is performed by calling the value_iteration function. 
    Within this function, there is a parameter q_function which is set to False by default. 
    This implies that the function will estimate V-values. Setting q_function to True during the function call switches the estimation to Q-values. 
    The function updates these values until the error becomes less than a specified threshold or until a maximum number of iterations (maxit) is reached. 
    During each iteration (an asynchronous update), either update_q_value or update_v_value is called depending on whether Q-values or V-values are 
    being estimated, respectively. In these functions, the Bellman equations are used for the calculation of values.

    The process of finding the optimal policy is conducted by calling generate_optimal_policy, which returns the policy as a dictionary:

        dict[position] = optimal_action 

    The q_function parameter also adjusts whether the policy is based on V-values or Q-values. 
    Inside this function, greedy_action is called for each state to derive the policy.

                                    Policy iteration:

    The policy_iteration function is used to execute the policy iteration algorithm. 
    Again, the q_function parameter determines whether the algorithm uses Q-values or V-values. 
    Within this function, a loop runs until two consecutive policies are the same. 
    Initially, it estimates the V-values or Q-values for the given policy, and then forms a greedy policy based on these estimated values. 
    This process repeats until convergence.
"""


def init_v_values(env: MazeEnvironment) -> dict:
    """
    Initializes the V-values for the value iteration algorithm.

    Args:
        env (MazeEnvironment): The maze environment.

    Returns:
        dict: A dictionary with positions as keys and initial V-values as values.
    """
    return {
        node.get_position(): -20 * random() if not env.is_terminal(node) else 0
        for node in env.get_graph()
    }


def init_q_values(env: MazeEnvironment) -> dict:
    """
    Initializes the Q-values for the Q-learning algorithm.

    Args:
        env (MazeEnvironment): The maze environment.

    Returns:
        dict: A dictionary with (position, action) tuples as keys and initial Q-values as values.
    """
    q_values = {}
    for node in env.get_graph():
        for action in ACTIONS:
            if env.is_terminal(node):
                q_values[node.get_position(), action] = 0
            else:
                q_values[node.get_position(), action] = -20 * random()
    return q_values


def update_v_value(
    env: MazeEnvironment, position: tuple, values: dict, gamma: float
) -> float:
    """
    Updates the V-value for a single node.

    Args:
        env (MazeEnvironment): The maze environment.
        position (tuple): The position of the node.
        values (dict): The current V-values.
        gamma (float): The discount factor.

    Returns:
        float: The updated V-value for the node.
    """
    node_at_position = env.get_current_pos_node(position)
    action_values = []  # Stores the calculated value for each action

    for action in ACTIONS:
        value_for_action = 0
        for next_node, transition_probability in env.get_action_probabilities(
            node_at_position, action
        ):
            future_value = values[next_node.get_position()]
            reward = next_node.get_reward()
            value_for_action += transition_probability * (reward + gamma * future_value)

        action_values.append(value_for_action)

    # Choose the highest value among all possible actions
    # If the node is a wall (all values are zero), return a specific value (e.g., -100)
    return max(action_values) if max(action_values) != 0 else -100


def update_q_value(
    env: MazeEnvironment, state: tuple, values: dict, gamma: float
) -> float:
    """
    Updates the Q-value for a single state-action pair.

    Args:
        env (MazeEnvironment): The maze environment.
        state (tuple): The state-action pair (position, action).
        values (dict): The current Q-values.
        gamma (float): The discount factor.

    Returns:
        float: The updated Q-value for the state-action pair.
    """
    node_at_state = env.get_current_pos_node(state[0])
    action_value_contributions = []

    for next_node, transition_prob in env.get_action_probabilities(
        node_at_state, state[1]
    ):
        future_values = [
            values[next_node.get_position(), next_action] for next_action in ACTIONS
        ]
        best_future_value = max(future_values)
        reward = next_node.get_reward()

        value_contribution = transition_prob * (reward + gamma * best_future_value)
        action_value_contributions.append(value_contribution)

    # Sum the contributions from all possible transitions
    # If no possible transitions (e.g., wall node), return a specific value (e.g., -100)
    return sum(action_value_contributions) if action_value_contributions else -100


def async_update_all_values(
    env: MazeEnvironment, values: dict, gamma: float, q_function: bool
) -> dict:
    """
    Performs an asynchronous update of all values (V or Q) for one iteration.

    Args:
        env (MazeEnvironment): The maze environment.
        values (dict): The current values (V or Q).
        gamma (float): The discount factor.
        q_function (bool): Whether to update Q-values instead of V-values.

    Returns:
        dict: The updated values after one iteration.
    """
    for state_action_pair in values:
        if q_function:
            # When updating Q-values, check if the current position is not terminal
            if not env.is_terminal_pos(
                state_action_pair[0]
            ):  # state_action_pair is (position, action)
                values[state_action_pair] = update_q_value(
                    env, state_action_pair, values, gamma
                )
        else:
            # When updating V-values, the state is just the position
            position = (
                state_action_pair  # Here, state_action_pair is actually just a position
            )
            if not env.is_terminal_pos(position):
                values[position] = update_v_value(env, position, values, gamma)

    return values


def value_iteration(
    env: MazeEnvironment,
    gamma: float,
    convergence_threshold: float,
    max_iterations: int = 100,
    use_q_function: bool = False,
) -> tuple:
    """
    Performs the value iteration algorithm on the maze environment.

    Args:
        env (MazeEnvironment): The maze environment on which the algorithm is run.
        gamma (float): The discount factor.
        convergence_threshold (float): The threshold for determining convergence.
        max_iterations (int, optional): Maximum number of iterations. Defaults to 100.
        use_q_function (bool, optional): Whether to use Q-function instead of value function. Defaults to False.

    Returns:
        tuple: A tuple containing the final values and the number of iterations it took to converge.
    """
    current_values = init_q_values(env) if use_q_function else init_v_values(env)
    for iteration_number in range(max_iterations):
        previous_values = copy(current_values)
        updated_values = async_update_all_values(
            env, current_values, gamma, use_q_function
        )
        max_error = max(
            abs(updated_values[state] - previous_values[state])
            for state in current_values
        )

        if max_error < convergence_threshold:
            return updated_values, iteration_number + 1

        current_values = updated_values

    return current_values, iteration_number + 1


def best_action_min_arg(actions_probs: list) -> int:
    """
    Determines the best action with the smallest number if there are multiple actions with the same value.
    *If values of both actions 1 and 2 are -1.0 , it returns 1*

    Args:
        actions_probs (list): A list of tuples (action, probability).

    Returns:
        int: The action with the smallest number among those with the highest value.
    """
    max_probability = max(prob for _, prob in actions_probs)
    max_probability_elements = [
        (action, prob) for action, prob in actions_probs if prob == max_probability
    ]

    min_action = min(action for action, _ in max_probability_elements)
    min_action_element = [
        (action, prob)
        for action, prob in max_probability_elements
        if action == min_action
    ][0][0]

    return min_action_element


def greedy_action(
    env: MazeEnvironment,
    current_node: Node,
    values: dict,
    gamma: float,
    use_q_values: bool = False,
) -> int:
    """
    Determines the best action to take from the current node based on a greedy approach.

    This function calculates the expected utility for each possible action from the current node and chooses the action that maximizes this expected utility. If 'use_q_values' is True, it uses Q-values for the calculation; otherwise, it uses V-values.

    Args:
        env (MazeEnvironment): The maze environment.
        current_node (Node): The current node from which the action is to be determined.
        values (dict): A dictionary containing the current estimated values (either V-values or Q-values).
        gamma (float): The discount factor.
        use_q_values (bool, optional): Specifies whether to use Q-values (True) or V-values (False) in the calculations. Defaults to False.

    Returns:
        int: The action that maximizes the expected utility. Returns None if no action is found (e.g., in a terminal state).
    """
    action_values = []
    for action in ACTIONS:
        expected_utility = 0
        for next_node, transition_prob in env.get_action_probabilities(
            current_node, action
        ):
            if use_q_values:
                future_values = [
                    values[next_node.get_position(), next_action]
                    for next_action in ACTIONS
                ]
                best_future_value = max(future_values)
            else:
                best_future_value = values[next_node.get_position()]

            expected_utility += transition_prob * (
                next_node.get_reward() + gamma * best_future_value
            )

        action_values.append((action, expected_utility))

    # Determine the best action based on the calculated utilities
    return best_action_min_arg(action_values) if action_values else None


def generate_optimal_policy(
    env: MazeEnvironment, values: dict, gamma: float, use_q_values: bool = False
) -> dict:
    """
    Generates the optimal policy for the maze environment based on the given value function or Q-values.

    The policy is a mapping from each node's position to the best action determined by a greedy approach,
    which maximizes the expected utility based on the current value estimates (V or Q values).

    Args:
        env (MazeEnvironment): The maze environment.
        values (dict): The dictionary of values (V or Q) used for determining the policy.
        gamma (float): The discount factor.
        use_q_values (bool, optional): Whether the values are Q-values. Defaults to False.

    Returns:
        dict: The optimal policy as a dictionary where keys are positions and values are actions.
    """
    optimal_policy = {}
    for node in env.get_graph():
        if not (node.is_terminal() or not node.is_steppable()):
            node_position = node.get_position()
            best_action = greedy_action(env, node, values, gamma, use_q_values)
            optimal_policy[node_position] = best_action

    return optimal_policy


def generate_random_policy(env: MazeEnvironment) -> dict:
    """
    Generates a random policy for the maze environment.

    Args:
        env (MazeEnvironment): The maze environment.

    Returns:
        dict: A dictionary representing the random policy, where keys are positions and values are actions.
    """
    policy = {}
    for s in env.get_graph():
        policy[s.get_position()] = rdm.choice(ACTIONS)
    return policy


def evaluate_values(
    env: MazeEnvironment,
    policy: dict,
    gamma: float,
    convergence_threshold: float,
    use_q_values: bool = False,
) -> dict:
    """
    Evaluates the values (V or Q) for a given policy.

    Args:
        env (MazeEnvironment): The maze environment.
        policy (dict): The policy to evaluate.
        gamma (float): The discount factor.
        convergence_threshold (float): The convergence threshold.
        use_q_values (bool, optional): Whether to evaluate Q-values instead of V-values. Defaults to False.

    Returns:
        dict: The evaluated values.
    """
    current_values = init_q_values(env) if use_q_values else init_v_values(env)
    updated_values = copy(current_values)

    while True:
        for state in current_values:
            node = env.get_current_pos_node(state[0] if use_q_values else state)

            if isinstance(node, WallNode):
                updated_values[state] = -100
            elif isinstance(node, TerminalNode):
                updated_values[state] = 0
            else:
                chosen_action = policy[node.get_position()]
                transition_probs = env.get_action_probabilities(node, chosen_action)

                if use_q_values:
                    update_state_value_q(
                        state, updated_values, transition_probs, gamma, policy
                    )
                else:
                    update_state_value_v(state, updated_values, transition_probs, gamma)

        max_error = max(
            abs(updated_values[s] - current_values[s]) for s in current_values
        )
        if max_error < convergence_threshold:
            return updated_values

        current_values = updated_values


def update_state_value_q(state, values, transition_probs, gamma, policy):
    """
    Helper function to update state value for Q-learning.
    """
    for next_node, prob in transition_probs:
        if isinstance(next_node, TerminalNode):
            values[state] = next_node.get_reward()
        else:
            next_action = policy[next_node.get_position()]
            values[state] = (
                next_node.get_reward()
                + gamma * values[next_node.get_position(), next_action]
            )


def update_state_value_v(state, values, transition_probs, gamma):
    """
    Helper function to update state value for value iteration.
    """
    value_sum = 0
    for next_node, prob in transition_probs:
        value_sum += prob * (
            next_node.get_reward() + gamma * values[next_node.get_position()]
        )
    values[state] = value_sum


def greedy_policy(env, values, gamma, q_function):
    """
    Generates a greedy policy based on the given values (V or Q).

    Args:
        env (MazeEnvironment): The maze environment.
        values (dict): The values (V or Q) based on which the policy is to be generated.
        gamma (float): The discount factor.
        q_function (bool): Whether the values are Q-values.

    Returns:
        dict: The generated greedy policy.
    """
    return generate_optimal_policy(env, values, gamma, q_function)


def policy_iteration(
    env: MazeEnvironment,
    gamma: float,
    convergence_threshold: float,
    use_q_values: bool = False,
) -> dict:
    """
    Executes the policy iteration algorithm to find the optimal policy in the given maze environment.

    The algorithm iteratively evaluates a policy and improves it based on the current value estimates until it converges to an optimal policy.

    Args:
        env (MazeEnvironment): The maze environment on which the algorithm is run.
        gamma (float): The discount factor.
        convergence_threshold (float): The threshold for determining convergence.
        use_q_values (bool, optional): Whether to use Q-function instead of value function. Defaults to False.

    Returns:
        dict: The optimal policy as a dictionary where keys are positions and values are actions.
    """
    current_policy = generate_random_policy(env)
    while True:
        value_estimates = evaluate_values(
            env, current_policy, gamma, convergence_threshold, use_q_values
        )
        improved_policy = greedy_policy(env, value_estimates, gamma, use_q_values)

        if improved_policy == current_policy:
            return current_policy

        current_policy = improved_policy


"""
    The algorithm can be tested with various maze dimensions, 
    but it's important to ensure that the number of actions is appropriately small. 
    For example, a 2x2 graph cannot have 4 actions, 
    as the implementation is designed such that each node connects to half of the other nodes. 
    Consequently, the number of actions must not exceed half of the total number of nodes. 
    This limitation does not apply to larger graphs.
"""


dims = (3, 3)
en = MazeEnvironment(dims)

v, v_it = value_iteration(en, 0.9, 0.01)
q, q_it = value_iteration(en, 0.9, 0.01, use_q_function=True)

# V values table
v_table = PrettyTable()
v_table.field_names = ["Position", "V Value"]
for position, value in v.items():
    v_table.add_row([position, value])

print(
    "\n----------------------------------- FINISHED VALUE ITERATION ALGORITHMS -----------------------------------\n"
)
print(f"Final V values on iteration {v_it}")
print(v_table)

# Q values table
q_table = PrettyTable()
q_table.field_names = ["State-Action", "Q Value"]
for state_action, value in q.items():
    q_table.add_row([state_action, value])

print(f"\nFinal Q values on iteration {q_it}")
print(q_table)

optimal_pol_v = generate_optimal_policy(en, v, 0.9)
optimal_pol_q = generate_optimal_policy(en, q, 0.9, use_q_values=True)

# Optimal policy after V iteration table
optimal_v_table = PrettyTable()
optimal_v_table.field_names = ["Position", "Optimal Action"]
for position, action in optimal_pol_v.items():
    optimal_v_table.add_row([position, action])

print(
    "\n---------------------------------- OPTIMAL POLICIES AFTER VALUE ITERATION ---------------------------------\n"
)
print("Optimal policy after V iteration is:")
print(optimal_v_table)

# Optimal policy after Q iteration table
optimal_q_table = PrettyTable()
optimal_q_table.field_names = ["Position", "Optimal Action"]
for position, action in optimal_pol_q.items():
    optimal_q_table.add_row([position, action])

print("\nOptimal policy after Q iteration is:")
print(optimal_q_table)

optimal_pol_pi_v = policy_iteration(en, 0.9, 0.01)
optimal_pol_pi_q = policy_iteration(en, 0.9, 0.01, use_q_values=True)

# Policy iteration using V table
pi_v_table = PrettyTable()
pi_v_table.field_names = ["Position", "Optimal Action"]
for position, action in optimal_pol_pi_v.items():
    pi_v_table.add_row([position, action])

print(
    "\n---------------------------------- OPTIMAL POLICIES AFTER POLICY ITERATION --------------------------------\n"
)
print("Optimal policy after policy iteration using V is:")
print(pi_v_table)

# Policy iteration using Q table
pi_q_table = PrettyTable()
pi_q_table.field_names = ["Position", "Optimal Action"]
for position, action in optimal_pol_pi_q.items():
    pi_q_table.add_row([position, action])

print("\nOptimal policy after policy iteration using Q is:")
print(pi_q_table)


en.print_graph()
plot_maze_graph(en)

# Zadatak level 3:
# Assume that Maze is graph-like and not grid-like, so that there are different possible actions to take from each cell.
# Add TELEPORT CELLS and modify the code to accomodate this kind of cells also.
# Implement value iteration algorithm using Q function instead of V function.
# Implement policy iteration algorithm using both V and Q functions.
# In all cases update, modify, and add visualization facilities to illustrate correctness of the implementation.

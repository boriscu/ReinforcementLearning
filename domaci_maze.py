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
    cvor grafa, dok su vrednosti kljuceva liste u kojima se nalaze manje liste koje sadrze
    akciju, naredni cvor i verovatnocu prelaska u taj cvor pri toj akciji:

        graph[node] = list[list[action, next_node, probability]]

    NAPOMENA: Ackije su prirodni brojevi, ako je akcija 0, to znaci da ne postoji veza
    izmedju ta da cvora ( svakako ce onda biti i verovatnoca 0 ).

    Kreiranjem MazeEnviorment objekta se generise nasumican graf kao atribut. Kao argument
    treba proslediti tuple(height, width) za visinu i sirinu lavirinta.

    Graf se inicijalizuje funkcijom initialize_graph (sama pozvana konstruktorom okruzenja).
    Unutar nje se formira nasumican graf, gde se za svaki cvor grafa generisu verovatnoce
    prelaska na sledece cvorove.Za terminalne cvorve i zidove, ne postoje stanja u koja se
    moze preci iz njih, za regularne postoji mogucnost prelaska na polovinu, dok za teleport
    posotji mogucnost na sve ostale cvorove (sem zidova). Detaljniji koraci su objasnjeni unutra.

    Metoda print_graph ispisuje graf, ali treba naglasiti da ona ne ispisuje tacan graf
    vec mnogo prirodnije:

        graph[node.get_position()] = list[list[action, next_node.get_position(), probability]

    Sto se tice ostalih metoda, jasno je sta rade po imenu. 

    Akcije su podesive, ali treba paziti da budu prirodni brojevi. Ogranicenje za njih je napisano
    kod poziva svih algoritama.
"""

ACTIONS = [1, 2, 3]


class MazeEnvironment:

    def __init__(self, dimensions: tuple[int, int]):
        self.graph_height = dimensions[0]
        self.graph_width = dimensions[1]
        self.graph = self.initialize_graph(self.graph_height, self.graph_width)

    def initialize_graph(self, width, height):
        graph = {}
        terminal = 0  # flag for terminal node

        # initialization of graph
        for w in range(1, width + 1):
            for h in range(1, height + 1):
                node = self.generate_random_node(w, h)
                graph[node] = []

                if isinstance(node, TerminalNode):
                    terminal += 1
        '''
            If there is no terminal node in random graph, it is manually made, but not from wall node
            because that would mean none of the other nodes would have probability different than zero
            to step on it ( which is characteristic of WallNode ).
        '''
        if not terminal:
            graph_list = list(graph)
            random_node = self.random_not_wall(graph_list)
            terminal_node = TerminalNode(-1, random_node.get_position()[0], random_node.get_position()[1])
            graph.pop(random_node)
            graph[terminal_node] = []

        # for all other nodes, probabilities are randomly generated
        for node in graph:
            graph[node] = self.set_probabilities(node, graph)

        return graph

    def set_probabilities(self, node, g):

        # terminal and wall node do not have any probabilities
        if isinstance(node, WallNode) or isinstance(node, TerminalNode):
            return []

        # list is easier to iterate, only because of that is node_list made
        nodes_list = [node for node in g]
        nodes_list_help = copy(nodes_list)  # only for iteration

        probabilities = []

        total_cells = self.graph_width * self.graph_height
        zero_cells = total_cells // 2  # number of cells with zero prob of stepping

        # probability is set to zero for wall modes
        for n in nodes_list_help:
            if isinstance(n, WallNode):
                probabilities.append([0, n, 0])
                zero_cells -= 1
                nodes_list.remove(n)

        # for regular nodes, half of probabilities should be zero, including wall nodes
        if isinstance(node, RegularNode):
            while zero_cells != 0:
                random_node = rdm.choice(nodes_list)
                nodes_list.remove(random_node)
                probabilities.append([0, random_node, 0])
                zero_cells -= 1

        # number of nodes that can be reached with one action
        # action_len = math.ceil(len(nodes_list) / len(ACTIONS))
        if len(nodes_list) / len(ACTIONS) > 1.5:
            action_len = math.ceil(len(nodes_list) / len(ACTIONS))
        else:
            action_len = math.floor(len(nodes_list) / len(ACTIONS))

        # shuffle actions
        random_actions = ACTIONS
        rdm.shuffle(random_actions)
        actions_copy = copy(random_actions)  # actions yet to be processed

        for action in random_actions:

            #  when it's the last action, all the left nodes are connected to the last action
            if len(actions_copy) == 1:
                nodes_for_current_action = nodes_list
            else:
                nodes_for_current_action = rdm.sample(nodes_list, action_len)
                for n in nodes_for_current_action:
                    nodes_list.remove(n)

            # generates probabilities with sum of 1
            non_zero_probabilities_action = self.generate_probabilities(nodes_for_current_action)

            # appends (action, next_node, prob)
            for i in range(len(nodes_for_current_action)):
                probabilities.append([action, nodes_for_current_action[i], non_zero_probabilities_action[i]])

            actions_copy.remove(action)  # removes processed action

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

    # prints graph (with position)
    def print_graph(self):
        values_graph = {}
        for node in self.graph:
            values_graph[node.get_position()] = []
            for [action, next_node, prob] in self.graph[node]:
                values_graph[node.get_position()].append([action, next_node.get_position(), prob])
        self.print_values(values_graph)
        return

    def print_values(self, g):
        print("\n----------------------------------------------- MAZE GRAPH "
              "------------------------------------------------ ")
        print(' ')
        for node in g:
            if self.get_current_pos_node(node).get_reward() == -10:
                print(node, "* : ", g[node])  # for regular node with penalty
                print(' ')
            else:
                print(node, ": ", g[node])
                print(' ')
        return

    # returns True if given node is terminal
    @staticmethod
    def is_terminal(node: Node):
        return node.is_terminal()

    # return True if node at position (x,y) is terminal
    def is_terminal_pos(self, position):
        for g in self.graph:
            if g.get_position() == position:
                return g.is_terminal()

    # returns graph
    def get_graph(self):
        return self.graph

    # returns node at position (x,y)
    def get_current_pos_node(self, pos):
        for g in self.graph:
            if g.get_position() == pos:
                return g
        raise Exception("Invalid position given.")

    # search for node that is not wall node in graph
    def random_not_wall(self, g):
        random_node = rdm.choice(g)
        if isinstance(random_node, WallNode):
            random_node = self.random_not_wall(g)
        return random_node

    # returns list[(next_node, prob)] for given action from given node
    def get_action_probabilities(self, node: Node, action):
        return [(pair[1], pair[2]) for pair in self.graph[node] if pair[0] == action]

    # returns next node for given probabilities, nodes_probs is list[node,prob]
    def get_next_node(self, nodes_probs):
        probabilities = [pair[1] for pair in nodes_probs]
        index = np.random.choice(len(probabilities), p=probabilities)
        return nodes_probs[index][0]


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


def plot_maze_graph(env: MazeEnvironment):
    g = nx.DiGraph()
    graph = env.get_graph()

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

    plt.show()


'''
    Sto se tice same implementacije algoritama, odradjena je na standardan nacin, u nastavku
    detljnije objasnjenje.

                                   Value iteration: 

        Procene Q i V vrednosti se vrse pozivom funkcije value_iteration, u okviru koje postoji
    parametar q_function koji je po default-u postaljen na False, sto znaci da se radi 
    procena V. Postavljanjem njega na True pri pozivu se racuna Q. Unutar funckije se vrsi 
    azuriranje vrednosti ili dok greska ne bude manja od zadate vrednosti ili maxit puta.
    Unutar asinhronog azuriranja ( jedne iteracije ) se poziva update_q_value/update_v_value
    u zavisnoti od potrebe, a u tim funkcijama subelmanove jednacine za racunanje vrednosti.

    Pronalazenje optimalne politike se vrsi pozivom generate optimal_policy, koja vraca politiku
    kao recnik:

        dict[position] = optimal_action 

    Takodje se parametrom q_function podesava da li je po V ili Q. Unutar nje se poziva 
    greedy_action za svako stanje i time dobija politika.

                                    Policy iteration:

        Pozivom funkcije policy_iteration se izvrsava algoritam, opet q_function parametar
    sluzi za biranje Q ili V funkcije. Unutar nje se izvrsava petlja dok se ne pojave 2 uzastupne
    iste politike, gde se prvo vrsi estimacija vrednosti V ili Q za zadatu politiku, a zatim se
    formira greedy politika za te zadate vrednosti i tako do kraja.

'''


# v values are dict[position] = value
def init_v_values(env: MazeEnvironment):
    return {s.get_position(): -20 * random() if not env.is_terminal(s) else 0 for s in env.get_graph()}


# q values are dict[position, action] = value
def init_q_values(env: MazeEnvironment):
    q = {}
    for s in env.get_graph():
        for action in ACTIONS:
            if env.is_terminal(s):
                q[s.get_position(), action] = 0
            else:
                q[s.get_position(), action] = -20 * random()
    return q


# update v value for one node
def update_v_value(env: MazeEnvironment, position, values, gamma):
    current_node = env.get_current_pos_node(position)
    possible_values = []
    for action in ACTIONS:
        x = 0
        for next_node, prob in env.get_action_probabilities(current_node, action):
            x += prob * (next_node.get_reward() + gamma * values[next_node.get_position()])
        possible_values.append(x)
    return max(possible_values) if max(possible_values) != 0 else -100  # wall node


# update q value for one node, action
def update_q_value(env: MazeEnvironment, state, values, gamma):
    current_node = env.get_current_pos_node(state[0])
    possible_values = []
    for next_node, prob in env.get_action_probabilities(current_node, state[1]):
        possible_values_one_node = []
        for action in ACTIONS:
            possible_values_one_node.append(values[next_node.get_position(), action])
        possible_values.append(prob * (next_node.get_reward() + gamma * max(possible_values_one_node)))
    return sum(possible_values) if possible_values else -100  # wall node


# updates values for one iteration
def async_update_all_values(env: MazeEnvironment, values, gamma, q_function):
    for s in values:
        if q_function:
            if not env.is_terminal_pos(s[0]):  # q values are (position,action)
                values[s] = update_q_value(env, s, values, gamma)
        else:
            if not env.is_terminal_pos(s):
                values[s] = update_v_value(env, s, values, gamma)
    return copy(values)


def value_iteration(env, gamma, eps, maxit=100, q_function=False):
    values = init_q_values(env) if q_function else init_v_values(env)
    for iteration in range(maxit):
        # print(f"\nIteration: {iteration + 1}")
        # print("Old values: ", values)
        values_copy = copy(values)
        new_values = async_update_all_values(env, values, gamma, q_function)
        # print("New values: ", new_values)
        err = max([abs(new_values[s] - values_copy[s]) for s in values])
        if err < eps:
            # print("Final error is: ", err)
            return new_values, iteration + 1
        # print("Error is: ", err)
        values = new_values
    return values, iteration + 1


# should return the smallest number action if there are more actions that have the same value
# if values of both actions 1 and 2 are -1.0 , it returns 1
def best_action_min_arg(actions_probs):
    max_probability = max(prob for _, prob in actions_probs)
    max_probability_elements = [(action, prob) for action, prob in actions_probs if prob == max_probability]

    min_action = min(action for action, _ in max_probability_elements)
    min_action_element = [(action, prob) for action, prob in max_probability_elements if action == min_action][0][0]

    return min_action_element


def greedy_action(env, current_node, values, gamma, q_function=False):
    if q_function:
        action_values = []
        for action in ACTIONS:
            for next_node, prob in env.get_action_probabilities(current_node, action):
                possible_values_one_node = []
                for next_action in ACTIONS:
                    possible_values_one_node.append(values[next_node.get_position(), next_action])
                action_values.append((action, prob * (next_node.get_reward() + gamma * max(possible_values_one_node))))
        return best_action_min_arg(action_values) if action_values else None
        # return max(action_values, key=lambda x: x[1])[0] if action_values else None
    else:
        action_values = []
        for action in ACTIONS:
            temp = 0
            for next_node, prob in env.get_action_probabilities(current_node, action):
                temp += prob * (next_node.get_reward() + gamma * values[next_node.get_position()])
            action_values.append((action, temp))
        return best_action_min_arg(action_values) if max(action_values) != 0 else None
        # return max(action_values, key=lambda x: x[1])[0] if max(action_values) != 0 else None


def generate_optimal_policy(env, values, gamma, q_function=False):
    return {
        node.get_position(): greedy_action(env, node, values, gamma, q_function)
        for node in env.get_graph()
        if not (node.is_terminal() or not node.is_steppable())
    }


def generate_random_policy(env):
    policy = {}
    for s in env.get_graph():
        policy[s.get_position()] = rdm.choice(ACTIONS)
    return policy


# does value iteration for given policy, to be used in policy iteration
def evaluate_values(env, policy, gamma, tolerance, q_function=False):
    values = init_q_values(env) if q_function else init_v_values(env)
    new_values = copy(values)
    if q_function:
        while True:
            # node_action is (s,a) tuple
            for node_action, prob in values:
                current_node = env.get_current_pos_node(node_action)
                if isinstance(current_node, WallNode):
                    new_values[node_action, prob] = -100
                elif isinstance(current_node, TerminalNode):
                    new_values[node_action, prob] = 0
                else:
                    action = policy[current_node.get_position()]
                    probs = env.get_action_probabilities(current_node, action)
                    next_node = env.get_next_node(probs)
                    if isinstance(next_node, TerminalNode):
                        new_values[node_action, prob] = next_node.get_reward()  # if next is terminal value is just -1
                    else:
                        next_action = policy[next_node.get_position()]
                        new_values[node_action, prob] = next_node.get_reward() + gamma * values[
                            next_node.get_position(), next_action]
            err = max([abs(values[s] - new_values[s]) for s in values])
            if err < tolerance:
                return new_values
            values = new_values
    else:
        while True:
            for node in env.get_graph():
                if isinstance(node, WallNode):
                    new_values[node.get_position()] = -100
                elif isinstance(node, TerminalNode):
                    new_values[node.get_position()] = 0
                else:
                    action = policy[node.get_position()]
                    probs = env.get_action_probabilities(node, action)
                    next_node = env.get_next_node(probs)
                    new_values[node] = next_node.get_reward() + gamma * values[next_node.get_position()]
            err = max([abs(values[s] - new_values[s]) for s in values])
            if err < tolerance:
                return new_values
            values = new_values


def greedy_policy(env, values, gamma, q_function):
    return generate_optimal_policy(env, values, gamma, q_function)


def policy_iteration(env: MazeEnvironment, gamma, tolerance, q_function=False):
    policy = generate_random_policy(env)
    while True:
        values = evaluate_values(env, policy, gamma, tolerance, q_function)
        new_policy = greedy_policy(env, values, gamma, q_function)
        if new_policy == policy:
            return policy
        policy = new_policy


'''
    Moze se isprobavati za razlicite dimenzije, ali paziti da akcija bude dovoljno malo
    jer ne moze na primer graf 2 puta 2 a 4 akcije, takva je implenetacija jer je naravljeno
    da svaki cvor ima prelaz na pola drugih cvorova. Odatle uslov da broj akcija ne sme biti
    veci od polovine ukupnog broja cvorova. Ogranicenje za vece grafove ne postoji.
'''

dims = (2, 3)

en = MazeEnvironment(dims)

v, v_it = value_iteration(en, 0.9, 0.01)
q, q_it = value_iteration(en, 0.9, 0.01, q_function=True)

print("\n----------------------------------- FINISHED VALUE ITERATION ALGORITHMS ----------------------------------- ")

print(f"\nFinal V values on iteration {v_it}")
print(v)
print(f"\nFinal Q values on iteration {q_it}")
print(q)

print("\n---------------------------------- OPTIMAL POLICIES AFTER VALUE ITERATION --------------------------------- ")

optimal_pol_v = generate_optimal_policy(en, v, 0.9)
print(f"\nOptimal policy after V iteration is:")
print(optimal_pol_v)

optimal_pol_q = generate_optimal_policy(en, q, 0.9, q_function=True)
print(f"\nOptimal policy after Q iteration is:")
print(optimal_pol_q)

print("\n---------------------------------- OPTIMAL POLICIES AFTER POLICY ITERATION -------------------------------- ")

optimal_pol_pi_v = policy_iteration(en, 0.9, 0.01)
print(f"\nOptimal policy after policy iteration using V is:")
print(optimal_pol_pi_v)

optimal_pol_pi_q = policy_iteration(en, 0.9, 0.01, q_function=True)
print(f"\nOptimal policy after policy iteration using Q is:")
print(optimal_pol_pi_q)

en.print_graph()
plot_maze_graph(en)

# Zadatak level 3:
# Assume that Maze is graph-like and not grid-like, so that there are different possible actions to take from each cell.
# Add TELEPORT CELLS and modify the code to accomodate this kind of cells also.
# Implement value iteration algorithm using Q function instead of V function.
# Implement policy iteration algorithm using both V and Q functions.
# In all cases update, modify, and add visualization facilities to illustrate correctness of the implementation.

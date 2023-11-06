from typing import Iterable
import random

import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange


class Bandit:
    """A bandit with uniform reward distribution."""

    def __init__(self, mean: float, span: float):
        """Initialize the bandit.

        Regardless of the received action, the bandit will return reward
        uniformly sampled from segment [`mean`-`span`, `mean`+`span`].

        Args:
            mean (float): Mean (expected) value of the reward.
            span (float): Span of the reward.
        """
        self.mean = mean
        self.span = span

    def pull_leaver(self) -> float:
        """Pull leaver and obtain reward.

        Returns:
            float: The obtained reward.
        """
        return self.mean + 2 * self.span * (
            random.random() - 0.5
        )  # random number in [mean-span, mean+span]


class BanditsEnvironment:
    """An environment consisting of multiple bandits."""

    def __init__(self, bandits: Iterable[Bandit], penalty=1000):
        """Initialize the environment.

        Args:
            bandits (iter[Bandit]): Bandits to be used within the environment.
            penalty (int, optional):
                If the external agents attempts to use a bandit not in the list,
                i.e. if the chosen action is negative or bigger than the index of
                the last bandit, the returned reward will be `-penalty`. Defaults to 1000.
        """
        self.bandits: list[Bandit] = list(bandits)
        self.penalty = penalty

    def take_action(self, a: int):
        """
        Select bandit `a` and pull its leaver.

        If the selected agent is valid, return the obtained reward.
        Otherwise, return negative penalty.
        """
        if a < 0 or a >= len(self.bandits):
            return -self.penalty
        else:
            return self.bandits[a].pull_leaver()


def choose_greedy_action(q):
    return np.argmax(q)


def choose_random_action(n):
    return random.randint(0, n - 1)


def choose_eps_greedy_action(q, eps):
    if random.random() > eps:
        return choose_greedy_action(q)
    else:
        return choose_random_action(len(q))


def train(bandits_no=5, attempts_no=5000, alpha=0.1, epsilon=0.1, plotting=True):
    bandits = [
        Bandit(10 * (random.random() - 0.5), 5 * random.random())
        for _ in range(bandits_no)
    ]
    env = BanditsEnvironment(bandits)

    q = [100 for _ in range(bandits_no)]
    rewards = []

    for t in trange(attempts_no):
        a = choose_eps_greedy_action(q, epsilon)
        r = env.take_action(a)
        q[a] = q[a] + alpha * (r - q[a])

        rewards.append(r)

    if plotting:
        plt.scatter(range(len(q)), q, marker=".")
        plt.scatter(range(len(q)), [b.mean for b in env.bandits], marker="x")

    return env.bandits, rewards, q


# Zadatak 1 : pokrenuti trening za razlicite vrednosti epsilon, prikazati rezultate i izvesti zakljucak o nagibu krive
def prvi_zadatak():
    plt.figure(figsize=(14, 3))
    for i in range(3):
        envi, rew, q1 = train(epsilon=10 ** (-i - 1), plotting=False)
        plot_e(3, i + 1, envi, rew)
        plt.title(10 ** (-i - 1))

    plt.show()


def plot_e(n, i, envi, rew):
    plt.subplot(1, n, i)
    g = np.cumsum(rew)
    max_r = max([b.mean for b in envi])
    plt.plot(g, "r")
    plt.plot(np.cumsum(max_r * np.ones(len(g))), "b")


# Zakljucak prvog zadatka jeste da bismo najbolje prosli (dobili najvecu kumulativnu nagradu) da smo koristili greedy politiku sve vreme.
# Prilikom smanjenja vrednosti epsilona, cesce je birana masina sa najvecom srednjom vrednosti (samim tim i veca nagarda) te je nagib
# krive kumulativne nagrade rastao.


# Zadatak 2 : Sa naucenim q i epsilon = 0, pustiti 100 iteracija
def drugi_zadatak():
    plt.figure(figsize=(8, 3))

    envi, rew, q1 = train(plotting=False)
    plot_e(2, 1, envi, rew)
    plt.title("5000 iteracija")

    plt.subplot(1, 2, 2)
    test_len = 100
    reward_q1 = [max(q1) for _ in range(test_len)]  # Uvek ce biti ista nagrada

    g1 = np.concatenate((rew, reward_q1))
    plot_e(2, 2, envi, g1)
    plt.title("5100 iteracija")
    plt.show()


# Zadatak 3, pod a) : nakon 4000 iteracija napraviti promenu srednje vrednosti i spanova bandita. Prikazati rezultate.

# Zadatak 3, pod b) : smisliti algoritam za nasumicnu promenu srednje vrednosti i spanova bandita, posle nekog vremena. Prikazati rezultate.


# Zadatak 4 : Uzeti 5 bandita i prikazati kako se njihove procenjene srednje vrednosti priblizavaju realnim tokom iteacija


def cetvrti_zadatak(attempts_no=5000, epsilon=0.1, alpha=0.1):
    bandits = [
        Bandit(10 * (random.random() - 0.5), 5 * random.random()) for _ in range(5)
    ]
    env = BanditsEnvironment(bandits)

    b1_mean = bandits[0].mean
    b2_mean = bandits[1].mean
    b3_mean = bandits[2].mean
    b4_mean = bandits[3].mean
    b5_mean = bandits[4].mean

    q = [100 for _ in range(5)]
    rewards = []
    b1 = []
    b2 = []
    b3 = []
    b4 = []
    b5 = []

    for t in trange(attempts_no):
        a = choose_eps_greedy_action(q, epsilon)
        r = env.take_action(a)
        q[a] = q[a] + alpha * (r - q[a])

        rewards.append(r)
        b1.append(q[0])
        b2.append(q[1])
        b3.append(q[2])
        b4.append(q[3])
        b5.append(q[4])

    plt.figure(figsize=(13, 8))

    plot_mean(b1_mean, b1, 1)
    plot_mean(b2_mean, b2, 2)
    plot_mean(b3_mean, b3, 3)
    plot_mean(b4_mean, b4, 4)
    plot_mean(b5_mean, b5, 5)

    plt.show()


def plot_mean(b_mean, b, indeks):
    plt.subplot(2, 3, indeks)
    plt.axhline(y=b_mean, color="r", linestyle="--", label=b_mean)
    plt.plot(b, "b")
    plt.legend()


prvi_zadatak()
drugi_zadatak()
cetvrti_zadatak()

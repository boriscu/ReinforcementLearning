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


# Prvi deo - razlicite vrednosti epsilon
def prvi_zadatak():
    plt.figure(figsize=(14, 3))

    plt.subplot(1, 3, 1)
    envi, rew, q1 = train(epsilon=0.1, plotting=False)
    g = np.cumsum(rew)
    max_r = max([b.mean for b in envi])
    plt.plot(g, "r")
    plt.plot(np.cumsum(max_r * np.ones(len(g))), "b")
    plt.title("Epsilon = 0.1")

    plt.subplot(1, 3, 2)
    envi, rew, q1 = train(epsilon=0.01, plotting=False)
    g = np.cumsum(rew)
    max_r = max([b.mean for b in envi])
    plt.plot(g, "r")
    plt.plot(np.cumsum(max_r * np.ones(len(g))), "b")
    plt.title("Epsilon = 0.01")

    plt.subplot(1, 3, 3)
    envi, rew, q1 = train(epsilon=0.001, plotting=False)
    g = np.cumsum(rew)
    max_r = max([b.mean for b in envi])
    plt.plot(g, "r")
    plt.plot(np.cumsum(max_r * np.ones(len(g))), "b")
    plt.title("Epsilon = 0.001")
    plt.show()


# Zakljucak prvog zadatka jeste da bismo najbolje prosli (dobili najvecu kumulativnu nagradu) da smo koristili greedy politiku sve vreme.


# Drugi deo - sa naucenim q i epsilon = 0, pustiti 100 iteracija
def drugi_zadatak():
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)

    envi, rew, q1 = train(plotting=False)
    g = np.cumsum(rew)
    max_r = max([b.mean for b in envi])
    plt.plot(g, "r")
    plt.plot(np.cumsum(max_r * np.ones(len(g))), "b")
    plt.title("5000 iteracija")

    plt.subplot(1, 2, 2)

    test_len = 100
    reward_q1 = [max(q1) for _ in range(test_len)]  # Uvek ce biti ista nagrada

    g1 = np.concatenate((rew, reward_q1))
    g = np.cumsum(g1)
    plt.plot(g, "r")
    plt.plot(np.cumsum(max_r * np.ones(len(g))), "b")
    plt.title("5100 iteracija")
    plt.show()


# Treci deo, pod a) - nakon 4000 iteracija promena srednje vrednosti i spanovi bandita.

# Treci deo, pod b) - smisliti algoritam za nasumicnu promenu srednje vrednosti i spana, posle nekog vremena


prvi_zadatak()
drugi_zadatak()

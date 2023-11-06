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


# Treci deo, pod a) - nakon 4000 iteracija promena srednje vrednosti i spanovi bandita.
def train_3a(bandits_no=5, attempts_no=5000, alpha=0.1, epsilon=0.1, plotting=True):
    bandits = [
        Bandit(10 * (random.random() - 0.5), 5 * random.random())
        for _ in range(bandits_no)
    ]
    env = BanditsEnvironment(bandits)

    q = [100 for _ in range(bandits_no)]
    rewards = []

    for t in trange(attempts_no):
        if t == 4000:
            # Prikaz stanja nakon 4000 iteracija
            plt.figure(figsize=(14, 3))
            plt.subplot(1, 3, 1)
            plt.scatter(range(len(q)), q, marker=".")
            plt.scatter(range(len(q)), [b.mean for b in env.bandits], marker="x")
            plt.title("Prvih 4000 iteracija")

            print(
                "Ukupno odstupanje procenjene srednje vrednosti od realne, posle 4000 iteracija: ",
                loss_function(q, env.bandits),
            )

            # Kreiranje novih (m, s)
            bandits = [
                Bandit(10 * (random.random() - 0.5), 5 * random.random())
                for _ in range(bandits_no)
            ]
            env = BanditsEnvironment(bandits)

        a = choose_eps_greedy_action(q, epsilon)
        r = env.take_action(a)
        q[a] = q[a] + alpha * (r - q[a])

        rewards.append(r)

    if plotting:
        # Prikaz stanja nakon izmene (m, s)
        plt.subplot(1, 3, 2)
        plt.scatter(range(len(q)), q, marker=".")
        plt.scatter(range(len(q)), [b.mean for b in env.bandits], marker="x")
        plt.title("Na kraju")

        print(
            "Ukupno odstupanje procenjene srednje vrednosti od realne, na kraju: ",
            loss_function(q, env.bandits),
        )

        # Pracenje nagrada kroz vreme
        plt.subplot(1, 3, 3)
        g = np.cumsum(rewards)
        plt.plot(g)
        plt.title("Nagrade tokom treninga")
        plt.show()

        plt.figure(figsize=(8, 3))
        plt.subplot(1, 2, 1)
        plt.plot(g[:1000], label="Prvih 1000 iteracija")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(g[-1500:], label="Poslednjih 1500 iteracija")
        plt.xticks([0, 750, 1500], [3500, 4250, 5000])
        plt.legend()

        plt.show()


def loss_function(q, envi):
    loss = 0
    m = [b.mean for b in envi]
    for a in range(len(q)):
        loss += abs(q[a] - m[a])
    return loss


# prvi_zadatak()
# drugi_zadatak()
train_3a(plotting=True)

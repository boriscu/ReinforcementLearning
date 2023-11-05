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

    def update_parameters(self, mean_shift: float, span_shift: float):
        """Updates the mean and span of the bandit's distribution with constraints.

        Args:
            mean_shift (float): Amount to shift the mean.
            span_shift (float): Amount to shift the span.
            min_span (float): Minimum allowed span.
            max_span (float, optional): Maximum allowed span. If None, no max limit is applied.
        """
        # Update mean and span
        self.mean += mean_shift
        self.span += span_shift

        self.mean = max(self.mean, 0)
        self.span = max(self.span, 0)


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


def time_change_train(
    bandits_no=5,
    attempts_no=5000,
    alpha=0.1,
    epsilon=0.1,
    mean_shift_interval=100,
    span_shift_interval=200,
    mean_shift_amount=1.5,
    span_shift_amount=1.5,
    shift_increment=4,
    num_runs=3,
    plotting=False,
):
    fig, axes = plt.subplots(
        2, num_runs, figsize=(5 * num_runs, 10), sharex="col", squeeze=False
    )

    for run in range(num_runs):
        bandits = [
            Bandit(10 * (random.random() - 0.5), 5 * random.random())
            for _ in range(bandits_no)
        ]
        env = BanditsEnvironment(bandits)

        q = [0 for _ in range(bandits_no)]
        rewards = []
        optimal_means = []
        actual_best_bandit = []
        estimated_best_bandit = []
        mean_shifts = []
        span_shifts = []

        for t in trange(
            attempts_no,
            desc=f"Run {run+1} with shifts {mean_shift_amount}, {span_shift_amount}",
        ):
            optimal_means.append(max(bandit.mean for bandit in env.bandits))
            actual_best_bandit.append(max(bandit.mean for bandit in env.bandits))
            estimated_best_bandit.append(env.bandits[np.argmax(q)].mean)

            if t % mean_shift_interval == 0:
                for bandit in env.bandits:
                    mean_shift = mean_shift_amount * (random.random() - 0.5) * 2
                    bandit.mean += mean_shift
                    mean_shifts.append(t)

            if t % span_shift_interval == 0:
                for bandit in env.bandits:
                    span_shift = span_shift_amount * random.random()
                    bandit.span += span_shift
                    span_shifts.append(t)

            a = choose_eps_greedy_action(q, epsilon)
            r = env.take_action(a)
            q[a] = q[a] + alpha * (r - q[a])

            rewards.append(r)

        # Plot cumulative rewards in the first row
        cumsum_ax = axes[0, run]
        cumsum_ax.plot(np.cumsum(rewards), label="Kumulativne Nagrade")
        cumsum_ax.plot(
            np.cumsum(optimal_means),
            label="Optimalne Kumulativne Nagrade",
            linestyle="--",
            alpha=0.75,
        )
        cumsum_ax.set_title(
            f"Pomeraji: Mean {mean_shift_amount}, Span {span_shift_amount}"
        )
        cumsum_ax.legend()

        # Plot actual vs estimated best bandit in the second row
        best_bandit_ax = axes[1, run]
        best_bandit_ax.plot(
            actual_best_bandit,
            label="Stvarni Najbolji Bandit",
            linestyle="-",
            alpha=0.5,
        )
        best_bandit_ax.plot(
            estimated_best_bandit,
            label="Procenjeni Najbolji Bandit",
            linestyle="--",
            alpha=0.5,
        )

        best_bandit_ax.scatter(
            mean_shifts,
            [actual_best_bandit[i] for i in mean_shifts],
            color="green",
            zorder=5,
            label="Mean Pomeraj",
            s=10,
        )
        best_bandit_ax.scatter(
            span_shifts,
            [actual_best_bandit[i] for i in span_shifts],
            color="red",
            zorder=5,
            label="Span Pomeraj",
            s=10,
        )

        best_bandit_ax.set_xlabel("Pokusaj")

        best_bandit_ax.legend()

        if run == 0:
            cumsum_ax.set_ylabel("Kumulativna Nagrada")
            best_bandit_ax.set_ylabel("Bandit Mean")

        mean_shift_amount += shift_increment
        span_shift_amount += shift_increment

    plt.tight_layout()
    if plotting:
        plt.show()


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
def treci_b_zadatak():
    time_change_train(plotting=True)


# prvi_zadatak()
# drugi_zadatak()
treci_b_zadatak()

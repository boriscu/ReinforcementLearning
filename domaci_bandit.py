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

    return env, rewards, q


# Zadatak 1 : pokrenuti trening za razlicite vrednosti epsilon, prikazati rezultate i izvesti zakljucak o nagibu krive
def prvi_zadatak():
    plt.figure(figsize=(14, 3))
    e = [0.95, 0.8, 0.5, 0.3, 0.1, 0.01, 0.001]
    iteration = 0
    for eps in e:

        envi, rew, q1 = train(epsilon=eps, plotting=False)

        iteration += 1

        plot_e(len(e), iteration, envi.bandits, rew)
        plt.title(eps)

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
def drugi_zadatak(plotting=False):
    env, rewards, q = train(plotting=False)
    learned_q = q.copy()

    # 100 iteracija sa epsilon = 0 (greedy metoda)
    greedy_rewards = []
    for _ in range(100):
        a = choose_greedy_action(learned_q)
        r = env.take_action(a)
        greedy_rewards.append(r)

    avg_rewards_eps_greedy = np.mean(rewards)
    avg_rewards_greedy = np.mean(greedy_rewards)
    print(f"Max Q naučena vrednost: {max(learned_q):.2f}")
    print("Prosečna nagrada - ε-Greedy Politika:", avg_rewards_eps_greedy)
    print("Prosečna nagrada - Čista Greedy Politika:", avg_rewards_greedy)
    if plotting:
        last_100_eps_greedy_rewards = rewards[-100:]

        cumulative_rewards_eps_greedy = np.cumsum(last_100_eps_greedy_rewards)
        cumulative_rewards_greedy = np.cumsum(greedy_rewards)

        plt.figure(figsize=(12, 6))
        plt.plot(
            cumulative_rewards_eps_greedy,
            label="Kumulativne Nagrade - Poslednjih 100 ε-Greedy",
        )
        plt.plot(
            cumulative_rewards_greedy,
            label="Kumulativne Nagrade - Greedy",
            linestyle="--",
        )

        plt.xlabel("Epoha")
        plt.ylabel("Kumulativna Nagrada")
        plt.title("Poredjenje Kumulativnih Nagrada: ε-Greedy vs Greedy")
        plt.legend()
        plt.show()

        # Poredjenje na poslednjih 10 nagrada
        max_mean_reward = max(bandit.mean for bandit in env.bandits)
        eps_greedy_every_10 = last_100_eps_greedy_rewards[::10]
        greedy_every_10 = greedy_rewards[::10]

        epochs = range(len(rewards) - 100, len(rewards), 10)
        plt.scatter(
            epochs,
            eps_greedy_every_10,
            label="ε-Greedy Nagrade (svaka 10ta)",
            marker="o",
            alpha=0.7,
        )

        plt.scatter(
            epochs,
            greedy_every_10,
            label="Greedy Nagrade (svaka 10ta)",
            marker="x",
            alpha=0.7,
        )

        plt.axhline(
            y=max_mean_reward,
            color="r",
            linestyle="--",
            label="Max Mean Moguća Nagrada",
        )

        plt.xlabel("Epoha")
        plt.ylabel("Nagrada")
        plt.title("Nagrada na 10 epoha za poslednjih 100 ε-Greedy i 100 Greedy")
        plt.legend()
        plt.show()


# Zadatak 3, pod a) - nakon 4000 iteracija promena srednje vrednosti i spanovi bandita.
def treci_zadatak_a(
        bandits_no=5, attempts_no=5000, alpha=0.1, epsilon=0.1, plotting=True
):
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
                "\nUkupno odstupanje procenjene srednje vrednosti od realne, posle 4000 iteracija: ",
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
            "\nUkupno odstupanje procenjene srednje vrednosti od realne, na kraju: ",
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


# Zadatak 3, pod b) - U random iteracijama promena srednjih vrednosti i spanova bandita.
def time_change_train(
        bandits_no=5,
        attempts_no=5000,
        alpha=0.1,
        epsilon=0.1,
        mean_shift_interval=100,
        span_shift_interval=200,
        shift_probability=0.5,
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
                desc=f"Run {run + 1} with shifts {mean_shift_amount}, {span_shift_amount}",
        ):
            optimal_means.append(max(bandit.mean for bandit in env.bandits))
            actual_best_bandit.append(max(bandit.mean for bandit in env.bandits))
            estimated_best_bandit.append(env.bandits[np.argmax(q)].mean)

            if t % mean_shift_interval == 0 and shift_probability > random.random():
                for bandit in env.bandits:
                    mean_shift = mean_shift_amount * (random.random() - 0.5) * 2
                    bandit.mean += mean_shift
                    mean_shifts.append(t)

            if t % span_shift_interval == 0 and shift_probability > random.random():
                for bandit in env.bandits:
                    span_shift = span_shift_amount * (random.random() - 0.5) * 2
                    bandit.span += span_shift
                    span_shifts.append(t)

            a = choose_eps_greedy_action(q, epsilon)
            r = env.take_action(a)
            q[a] = q[a] + alpha * (r - q[a])

            rewards.append(r)

        # Prikaz kumulativnih nagrada u prvom redu
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

        # Prikaz stvarnog naspram procenjenog najboljeg bandita u drugom redu
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


def treci_zadatak_b(plotting=False):
    time_change_train(plotting=plotting)


# Zakljucak treceg zadatka
# Povećanje pomeraja srednjih vrednosti i raspona čini da je algoritmu teže da prati optimalne nagrade.
# To je posebno izraženo u kasnijim iteracijama,
# što ukazuje na to da algoritam ima poteškoće u prilagođavanju kada se promene događaju češće ili su značajnije.
# Sa povećanjem pomeraja, algoritam češće greši u identifikaciji stvarno najboljeg bandita.
# To može biti posledica toga što algoritam teži da više veruje procenama zasnovanim na prethodnim iskustvima,
# koje postaju manje pouzdane kako se promene nagrada bandita čine češće.
# Uprkos izazovima u praćenju najboljeg bandita,
# algoritam ipak pokazuje sposobnost da identifikuje novog najboljeg bandita nakon promena.
# Vreme potrebno algoritmu da pronađe novog najboljeg bandita može biti pokazatelj njegove efikasnosti u učenju i adaptaciji.


# Zadatak 4 : Uzeti 5 bandita i prikazati kako se njihove procenjene srednje vrednosti priblizavaju realnim tokom iteacija
def cetvrti_zadatak(attempts_no=5000, epsilon=0.1, alpha=0.1):
    bandits = [
        Bandit(10 * (random.random() - 0.5), 5 * random.random()) for _ in range(5)
    ]
    env = BanditsEnvironment(bandits)

    q = [100 for _ in range(5)]
    rewards = []
    # Matrica 5 bandita x promene q za svaki
    q_value_history = np.zeros((attempts_no, 5))

    for t in trange(attempts_no):
        a = choose_eps_greedy_action(q, epsilon)
        r = env.take_action(a)
        q[a] = q[a] + alpha * (r - q[a])

        rewards.append(r)
        q_value_history[t] = q

    plot_mean(bandits, q_value_history)

    plt.show()


def plot_mean(bandits, q_value_history):
    plt.figure(figsize=(13, 8))
    for i in range(5):
        plt.subplot(2, 3, i + 1)
        plt.axhline(
            y=bandits[i].mean,
            color="r",
            linestyle="--",
            label=f"Realna Srednja vrednost: {bandits[i].mean:.2f}",
        )
        plt.plot(q_value_history[:, i], label=f"Procenjena srednja vrednost")
        plt.legend()
        plt.title(f"Bandit {i + 1}")


# Dodatni zadatak: Prikazati realne i procenjene vrednosti za sve bandite na jednom grafiku
def dodatni_zadatak(
        bandits_no=5, attempts_no=5000, alpha=0.1, epsilon=0.1, step=500, plotting=True
):
    bandits = [
        Bandit(10 * (random.random() - 0.5), 5 * random.random())
        for _ in range(bandits_no)
    ]
    env = BanditsEnvironment(bandits)
    q = [0 for _ in range(bandits_no)]

    q_value_history = np.zeros((attempts_no, len(bandits)))

    bandits_mean_history = np.zeros((attempts_no, len(bandits)))
    bandits_mean_history[0] = [bandit.mean for bandit in bandits]

    for t in trange(attempts_no):
        if t % step == 0:

            # Kreiranje novih bandita (m, s)
            bandits = [
                Bandit(10 * (random.random() - 0.5), 5 * random.random())
                for _ in range(bandits_no)
            ]
            env = BanditsEnvironment(bandits)
            bandits_mean_history[t] = [bandit.mean for bandit in bandits]

        a = choose_eps_greedy_action(q, epsilon)
        r = env.take_action(a)
        q[a] = q[a] + alpha * (r - q[a])

        if t != 0 and t % step != 0:
            bandits_mean_history[t] = bandits_mean_history[t-1]

        q_value_history[t] = q

    plot_all_bandits_history(bandits, q_value_history, bandits_mean_history)

# Primeti se da sto je parametar step manji, algoritmu je teze tj. losije ce procenjivati realne vrednosti
# zato sto nema dovoljno iteracija za preciznu procenu.

def plot_all_bandits_history(bandits, q_value_history, bandtis_hist):
    plt.figure(figsize=(15, 10))
    for i in range(len(bandits)):
        plt.plot(bandtis_hist[:, i], label=f"Realna srednja vrednost bandita {i}", linestyle='--')
        plt.plot(q_value_history[:, i], label=f"Procenjena srednja vrednost bandita {i}")
        plt.legend()
    plt.show()


'''
prvi_zadatak()
drugi_zadatak(plotting=True)
treci_zadatak_a(plotting=True)
treci_zadatak_b(plotting=True)
cetvrti_zadatak()
'''
dodatni_zadatak()

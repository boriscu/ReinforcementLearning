"""
Level 3
Make the game symmetric in the following way: Instead of the Player and the Dealer, let us talk about Player 1 and Player 2.

Each game consists of two rounds, in the first round Player 1 plays first and in the second round Player 2 plays first.

Each round is a single game of blackjack with one card handled to each player and one common card drawn (so that each player forms the 
initial sum from two cards, with one card being shared by both players). Each player plays (chooses HIT or HOLD) in each turn. If one of 
the players becomes busted, or if one of the player chooses HOLD, the other player continues drawing cards until he/she is also busted 
or chooses HOLD. At the end, the player with highest sum less than 21 wins the round.

Upon completion of the round, each player receives +1, 0, or -1, depending on the outcome. The total outgome of one game is the sum of 
outcomes of the two rounds.
"""
from enum import Enum
from dataclasses import dataclass, astuple
from typing import Callable
from random import random, randint
from copy import deepcopy
import matplotlib.patches as mpatches

from tqdm import trange
from rich import print

import numpy as np
import matplotlib.pyplot as plt


class CardSuit(Enum):
    """An enumeration representing card suits."""

    DIAMONDS = 0
    CLUBS = 1
    HEARTS = 2
    SPADES = 3

    def __repr__(self):
        match self:
            case CardSuit.DIAMONDS:
                return "♦"
            case CardSuit.CLUBS:
                return "♣"
            case CardSuit.HEARTS:
                return "♥"
            case CardSuit.SPADES:
                return "♠"


class CardValue(Enum):
    """An enumeration representing card values."""

    ACE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 12
    DAME = 13
    KING = 14

    def __repr__(self):
        if self == CardValue.ACE:
            return "A"
        elif self.value >= 2 and self.value <= 10:
            return str(self.value)
        elif self == CardValue.JACK:
            return "J"
        elif self == CardValue.DAME:
            return "D"
        elif self == CardValue.KING:
            return "K"


@dataclass
class Card:
    """A playing card."""

    value: CardValue
    suit: CardSuit

    def __repr__(self):
        return f"{repr(self.value)}{repr(self.suit)}"


class CardDeck:
    """A deck of cards.

    The deck contains 52 playing cards, which are shuffled and drawn consecutively.
    Once all cards have been drawn, the deck reshuffles automatically.
    """

    def reshuffle(self):
        """Refill the deck with all playing cards and reshuffle it."""
        self.cards = self.multiplicity * [
            Card(value=v, suit=s) for v in iter(CardValue) for s in iter(CardSuit)
        ]
        np.random.shuffle(self.cards)

    def __init__(self, multiplicity=5):
        """Create a shuffled deck of cards.

        Args:
            multiplicity (int): The number of 52-card sets to put inside a deck.
        """
        self.multiplicity = multiplicity
        self.reshuffle()

    def draw(self):
        """Draw a card from the deck.

        Once drawn, the card is removed from the deck. Once there are no more cards inside,
        the deck will automatically refill and reshuffle.
        """
        if not self.cards:
            self.reshuffle()
        return self.cards.pop(0)


@dataclass
class State:
    """The state of a Blackjack game"""

    total: int
    has_ace: bool
    dealer_total: int

    def __hash__(self):
        return hash(astuple(self))

    def __repr__(self):
        return repr(astuple(self))


def update_total(total: int, has_ace: bool, card_value: CardValue) -> tuple[int, bool]:
    """Update (a portion of) the player state by taking account a newly drawn card.

    Given the information regarding the current player state (total and ACE info), and the
    value of a newly drawn card, updates the current player state (both total and ACE info).

    The updated total returned by this function can be bigger than 21.
    The function correctly tracks statuses of ACEs (i.e. it changes values of ACEs from
    11 to 1, when appropriate, in order for the total not to excede 21), however, if
    after this the total still remains bigger than 21 the function will simply return it
    as is.

    Args:
        total (int): player total
        has_ace (bool): indicates if the player has usable ACE
        card_value (CardValue): value of the newly drawn card

    Returns:
        int: updated player total
        bool: updated info indicating if the player has a usable ACE
    """
    if card_value == CardValue.ACE:
        if total + 11 <= 21:
            # We have an ACE and the total is less than 21.
            # We can keep the ACE as 11.
            # Notice that it was impossible that we had another ACE in posession
            # before (since in that case the total would surely be bigger than 21)
            return (total + 11, True)
        else:
            # The total excedes 21, and we must count the ACE as 1.
            # It is possible we had another active ACE before, if so the
            # status of this old ACE does not change.
            return (total + 1, has_ace)
    else:
        total += min(card_value.value, 10)
        if total > 21:
            # The player is potentially busted.
            if has_ace:
                # If we have an active ACE, we are saved.
                # The ACE value is reduced from 11 to 1 (and the total is reduced by 10).
                # However, the cost is that we no longer have an active ACE.
                total -= 10
                return (total, False)
            else:
                # We have not active ACE, so we cannote decrease the total
                return (total, has_ace)
        # The total is less than 21, we may return the state as is.
        return (total, has_ace)


def random_state() -> State:
    """Generate a random state."""
    total = randint(2, 21)
    if total >= 11:
        r = random()
        if r >= 0.5:
            has_ace = True
        else:
            has_ace = False
    else:
        has_ace = False
    dealer_value = randint(2, 11)
    return State(total, has_ace, dealer_value)


def get_init_state(
    state: State | str | None,
    deck: CardDeck,
    common_card: Card | None,
    debug: bool = False,
) -> State:
    """Generate initial state."""
    if state is None:
        player_card = deck.draw()
        total, has_ace = update_total(0, False, player_card.value)
        if common_card is None:
            raise ValueError("Common card cannot be None")

        common_card_value = min(common_card.value.value, 10)
        if debug:
            print(f"Player initial hand: {player_card}, {common_card}")
        return State(total, has_ace, common_card_value)
    else:
        if isinstance(state, str):
            assert state == "random", f"Invalid init state specification: '{state}'"
            return random_state()
        else:
            return state


def get_init_states(deck: CardDeck, debug: bool = False) -> (State, State):
    # Draw the common card and print it
    common_card = deck.draw()

    # Initialize state for Player 1
    player1_state = get_init_state(None, deck, common_card, debug)
    player1_state.total, player1_state.has_ace = update_total(
        player1_state.total, player1_state.has_ace, common_card.value
    )

    # Initialize state for Player 2
    player2_state = get_init_state(None, deck, common_card, debug)
    player2_state.total, player2_state.has_ace = update_total(
        player2_state.total, player2_state.has_ace, common_card.value
    )
    return player1_state, player2_state


def all_states() -> list[State]:
    """Create a list of all possible Blackjack states."""
    states = []
    for total in range(2, 22):
        for dealer_value in range(2, 12):
            states.append(State(total, False, dealer_value))
            if total >= 11:
                states.append(State(total, True, dealer_value))
    return states


class Action(Enum):
    """Blackjack action."""

    HIT = 0
    HOLD = 1

    def __repr__(self):
        return "HIT" if self == Action.HIT else "HOLD"


def random_action() -> Action:
    """Return random action."""
    r = random()
    if r <= 0.5:
        return Action.HIT
    else:
        return Action.HOLD


def random_policy(s: State) -> Action:
    return random_action()


def dealer_policy(state: State) -> Action:
    """Dealer policy: Hit if below 17, Hold if 17 or above."""
    return Action.HIT if state.total < 17 else Action.HOLD


Policy = Callable[[State], Action]


def get_init_action(
    action: Action | str | None, state: State, policy: Policy
) -> Action:
    """Generate initial action."""
    if action is not None:
        if isinstance(action, str):
            assert action == "random", f"Ivalid init action specification: '{action}'"
            return random_action()
        else:
            return action
    else:
        return policy(state)


TurnLog = list[tuple[State, Action]]
ReportCallback = Callable[[str], None]


def play_turn(
    policy: Policy,
    deck: CardDeck,
    init_state: State | None = None,
    init_action: Action | None = None,
    report_callback: ReportCallback | None = None,
) -> tuple[int | None, TurnLog]:
    """A single playing turn of an agent.

    Args:
        policy (Policy): decision policy used by the agent
        deck (CardDeck): deck of cards from which to draw
        init_state (State | None): initial state or `None` if the initial state
                                   is to be selected by drawing from the deck.
        init_action (Action | None): initial action or `None` if the initial action
                                     is to be selected according to the decision policy
        report_callback (ReportCallback | None): callback used to report progress
                                                 if `None`, the turn will be played silently

    Return:
        int | None: The final agent's total or `None` if the agent is busted
        TurnLog: sequence of states and actions observed during gameplay
    """
    report = lambda txt: report_callback and report_callback(txt)
    state = get_init_state(init_state, deck, None)
    action = get_init_action(init_action, state, policy)
    total, has_ace, dealer_value = astuple(state)
    report(f"[bold]initial state[/] {state}")
    report(
        f"initial total {total} with ACE {has_ace} => [bold]initial action[/] {action}"
    )
    turn_log = [(state, action)]
    while action == Action.HIT:
        card = deck.draw()
        total, has_ace = update_total(total, has_ace, card.value)
        state = State(total, has_ace, dealer_value)
        if total > 21:
            # Since total is above 21, the state is not valid and should not
            # be logged!
            report(
                f"turned finished - [bold]BUSTED![/] card drawn {card} => total {total} with ACE {has_ace}"
            )
            return None, turn_log
        else:
            action = policy(state)
            turn_log.append((state, action))
            report(
                f"card drawn {card} => total {total} with ACE {has_ace} => action chosen {action}"
            )
    report(f"turn finished with final total {total}")
    return total, turn_log


Experience = tuple[State, Action, float]


def compare_totals(player1_total, player2_total) -> int:
    # Check for busts
    player1_bust = player1_total is None or player1_total > 21
    player2_bust = player2_total is None or player2_total > 21

    # Both players bust
    if player1_bust and player2_bust:
        return 0  # Draw

    # Only Player 1 busts
    if player1_bust:
        return -1  # Player 2 wins

    # Only Player 2 busts
    if player2_bust:
        return 1  # Player 1 wins

    # Compare totals if no busts
    if player1_total > player2_total:
        return 1  # Player 1 wins
    elif player1_total < player2_total:
        return -1  # Player 2 wins
    else:
        return 0  # Draw


def compute_gain(rewards: list[float], gamma: float) -> float:
    """
    Compute the total gain given the list of future rewards.

    Args:
        rewards (list[float]): List of future rewards.
        gamma: discount factor

    Return:
        float: The total gain
    """
    g = 0
    w = 1
    for r in rewards:
        g += w * r
        w *= gamma
    return g


def build_experience(
    player_log: TurnLog, opponent_log: TurnLog, result: int, gamma: float
) -> Experience:
    """
    Compute experience from the turn logs and the final result for a two-player game.

    Args:
        player_log (TurnLog): List of state-action pairs for the player.
        opponent_log (TurnLog): List of state-action pairs for the opponent.
        result (int): Final result of the game from the player's perspective.
        gamma (float): Discount factor.

    Return:
        Experience: List of state-action-reward-next_state-next_action tuples for the player.
    """
    player_experience = []
    for i in range(len(player_log)):
        state, action = player_log[i]
        # Reward is 0 for all but the last state, where it is the game result
        reward = 0 if i < len(player_log) - 1 else result
        next_state, next_action = (
            player_log[i + 1] if i + 1 < len(player_log) else (None, None)
        )
        player_experience.append((state, action, reward, next_state, next_action))

    return player_experience


def play_game(
    policy1: Policy,
    policy2: Policy,
    deck: CardDeck,
    gamma: float = 1.0,
    game_report_callback: ReportCallback | None = None,
    player1_report_callback: ReportCallback | None = None,
    player2_report_callback: ReportCallback | None = None,
    debug: bool = False,
) -> tuple[int, list, list]:
    """
    Simulate a two-player game with specified policies for each player and a card deck.

    This function executes a game of Blackjack (or similar) where two players take turns based
    on their respective policies. After the game,
    it generates experiences for each player suitable for reinforcement learning algorithms SARSA and Q-Learning.

    Args:
        policy1 (Policy): The policy function to be used by Player 1.
        policy2 (Policy): The policy function to be used by Player 2.
        deck (CardDeck): The deck of cards used for the game.
        gamma (float): Discount factor for calculating returns.
        game_report_callback (ReportCallback, optional): Callback function for game-level reporting.
        player1_report_callback (ReportCallback, optional): Callback function for reporting Player 1's actions.
        player2_report_callback (ReportCallback, optional): Callback function for reporting Player 2's actions.
        debug (bool, optional): Flag to enable debug mode for additional output.

    Returns:
        tuple[int, list, list]: A tuple containing the game result (int), and two lists of experiences
                                 (state, action, reward, next_state, next_action) for Player 1 and Player 2.
    """

    report = lambda txt: game_report_callback and game_report_callback(txt)

    # Initialize states for both players
    player1_state, player2_state = get_init_states(deck, debug)

    # Player 1 takes their turn
    player1_total, player1_log = play_turn(
        policy1, deck, player1_state, report_callback=player1_report_callback
    )

    # Player 2 takes their turn
    player2_total, player2_log = play_turn(
        policy2, deck, player2_state, report_callback=player2_report_callback
    )

    game_result = compare_totals(player1_total, player2_total)
    report(f"Game result: {game_result}")

    player1_experience = build_experience(player1_log, player2_log, game_result, gamma)
    player2_experience = build_experience(
        player2_log, player1_log, -game_result, gamma
    )  # Note the negative result for the second player

    return game_result, player1_experience, player2_experience


@dataclass
class PolicyTestingResult:
    games_no: int
    score: int
    wins: float
    draws: float
    losses: float

    def loss_per_game(self):
        """Average loss per game."""
        return self.score / self.games_no

    def __repr__(self):
        return f"total score={self.score}/{self.games_no} games ({self.loss_per_game():.4f} per game) :: W {self.wins*100:.2f}% | D {self.draws*100:.2f}% | L {self.losses*100:.2f}%"


def apply_policy_exhaustive(
    policy1: Policy, policy2: Policy, deck: CardDeck, epoch_no=5, gamma=1
) -> tuple[PolicyTestingResult, PolicyTestingResult]:
    wins1, losses1, draws1, score1, games_no1 = 0, 0, 0, 0, 0
    wins2, losses2, draws2, score2, games_no2 = 0, 0, 0, 0, 0

    for _ in range(epoch_no):
        res, exp_player1, exp_player2 = play_game(policy1, policy2, deck, gamma=gamma)

        # Update scores and counts for Player 1
        games_no1 += 1
        if res == 1:
            wins1 += 1
        elif res == 0:
            draws1 += 1
        else:
            losses1 += 1
        score1 += res

        # Update scores and counts for Player 2
        games_no2 += 1
        if res == -1:
            wins2 += 1
        elif res == 0:
            draws2 += 1
        else:
            losses2 += 1
        score2 -= res  # Negative because a win for Player 1 is a loss for Player 2

    result1 = PolicyTestingResult(
        games_no1, score1, wins1 / games_no1, draws1 / games_no1, losses1 / games_no1
    )
    result2 = PolicyTestingResult(
        games_no2, score2, wins2 / games_no2, draws2 / games_no2, losses2 / games_no2
    )

    return result1, result2


GainsDict = dict[State, tuple[list[float], list[float]]]


def create_gains_dict(experience: Experience) -> GainsDict:
    """Create gains dict from the given experience."""
    gains_dict = dict()
    for s, a, g in experience:
        action_gains = gains_dict.get(s, ([], []))
        action_gains[a.value].append(g)
        gains_dict[s] = action_gains
    return gains_dict


def update_q_value(current: float, target: float, alpha: float) -> float:
    """Update the given current estimate given the new target.

    Args:
        current (float): existing estimate
        target (float): the given target
        alpha (float): learning rate

    Return:
        float: the new, updated estimate
    """
    if current is not None and target is not None:
        return current + alpha * (target - current)
    elif current is None:
        return target
    elif target is None:
        return current
    else:
        return None


QDict = dict[State, tuple[float, float]]


def create_greedy_policy(q_dict: QDict) -> Policy:
    """Create a greedy policy function based on the given Q-dictionary."""
    no_actions = len(list(Action))

    def policy(s: State):
        q_values = q_dict.get(s, None)
        if q_values is not None:
            assert len(q_values) == no_actions, f"Invalid Q-dict for state {s}."
            if any([q_value is None for q_value in q_values]):
                return random_policy(s)
            else:
                ndx = np.argmax(q_values)
                return Action(ndx)
        else:
            return random_policy(s)

    return policy


def update_Q(
    q_dict: QDict, experience: Experience, alpha=0.1, gamma=0.9, method="qlearning"
):
    """
    Update the Q-function based on the given experience using either the SARSA or Q-Learning
    algorithm. This function iterates over a list of experiences, updating the Q-values
    for the actions taken in those experiences according to the chosen method.

    Q-Learning Update Formula:
    Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a))
    Where:
    - Q(s, a) is the current Q-value of state-action pair (s, a).
    - reward is the reward received after executing action a in state s.
    - gamma is the discount factor, weighing the importance of future rewards.
    - max(Q(s', a')) is the maximum Q-value for the next state s' across all possible actions a'.
    - alpha is the learning rate, determining the impact of the new information.

    SARSA Update Formula:
    Q(s, a) = Q(s, a) + alpha * (reward + gamma * Q(s', a') - Q(s, a))
    Where:
    - Q(s', a') is the Q-value of the action a' taken in the next state s'.
    - Other variables retain their meaning as described in the Q-Learning formula.

    Args:
        q_dict (QDict): A dictionary mapping state-action pairs to Q-values.
        experience (Experience): A list of tuples containing the experiences to be processed.
                                 Each tuple consists of (state, action, reward, next_state, next_action).
        alpha (float): The learning rate, a factor that determines how much the newly acquired
                       information will override the old information.
        gamma (float): The discount factor, used to balance the importance of immediate and future rewards.
        method (str): Specifies the learning method to use for updating Q-values. It can be
                      either 'sarsa' for the SARSA algorithm or 'qlearning' for Q-Learning.

    Returns:
        QDict: The updated Q-values dictionary after processing all experiences.

    Note:
        - This function updates the Q-values in-place within the passed `q_dict`.
        - The choice between SARSA and Q-Learning affects how future rewards are estimated and
          incorporated into the current Q-value estimates. SARSA considers the actual next action
          (a'), leading to a more conservative approach compared to Q-Learning, which optimistically
          estimates future rewards based on the maximum possible Q-value for the next state (s').
    """
    for state, action, reward, next_state, next_action in experience:
        old_value = q_dict.get(state, (0, 0))[action.value]

        if next_state is not None:
            if method == "sarsa":
                next_q_value = q_dict.get(next_state, (0, 0))[next_action.value]
                target = reward + gamma * next_q_value
            elif method == "qlearning":
                next_q_values = q_dict.get(next_state, (0, 0))
                target = reward + gamma * max(next_q_values)
        else:
            target = reward

        new_value = update_q_value(old_value, target, alpha)
        q_values = list(q_dict.get(state, (0, 0)))
        q_values[action.value] = new_value
        q_dict[state] = tuple(q_values)

    return q_dict


def visualize_policy(policy):
    """Visualize the policy.

    The policy will be visualized using a colored greed.
    The horizontal axis will correspond to the player total and the vertical axis to the
    dealer's total.

    If a cell is colored in red, it means that the action is HIT, regardless of existence of usable ACE.
    If a cell is colored in blue, it means HOLD, regardless of existence of usable ACE.
    Green cells are those in which HIT will be played only if there is a usable ACE.
    Black cells are those in which HIT will be played only if there is no usable ACE.

    Since the last case (play HIT if there is no usable ACE, and HOLD otherwise) seems counterintutive,
    the color black is selected to distinguish it from other states.
    """
    player_values = list(range(2, 21))
    dealer_values = list(range(2, 12))
    board = np.ones(shape=(len(dealer_values), len(player_values), 3), dtype=np.uint8)
    for r, dv in enumerate(dealer_values):
        for c, pv in enumerate(player_values):
            if pv < 11:
                action_t = policy(State(pv, False, dv)) == Action.HIT
                if action_t:
                    board[r, c, :] = (255, 0, 0)
                else:
                    board[r, c, :] = (0, 0, 255)
            else:
                action_t = policy(State(pv, True, dv)) == Action.HIT
                action_f = policy(State(pv, False, dv)) == Action.HIT
                if action_t and action_f:
                    board[r, c, :] = (255, 0, 0)
                elif action_t and not action_f:
                    board[r, c, :] = (0, 255, 0)
                elif not action_t and not action_f:
                    board[r, c, :] = (0, 0, 255)
                else:
                    board[r, c, :] = (0, 0, 0)
    plt.imshow(board, extent=[2, 21, 12, 2])
    plt.xticks(np.arange(2.5, 21.5, 1), np.arange(2, 21, 1))
    plt.yticks(np.arange(2.5, 12.5, 1), np.arange(2, 12, 1))
    plt.xlabel("player total")
    plt.ylabel("oponent total")

    # Create patches for the legend
    red_patch = mpatches.Patch(color="red", label="HIT (with or without ACE)")
    blue_patch = mpatches.Patch(color="blue", label="HOLD (with or without ACE)")
    green_patch = mpatches.Patch(color="green", label="HIT only with usable ACE")
    black_patch = mpatches.Patch(color="black", label="HIT only without usable ACE")

    # Add the legend to the plot
    plt.legend(handles=[red_patch, blue_patch, green_patch, black_patch])

    plt.show()


def pre_train_players(deck, num_games, alpha=0.1, gamma=0.9, method: str = "qlearning"):
    q_dict_player1 = dict()
    q_dict_player2 = dict()

    for _ in trange(num_games, desc="Pretraining"):
        # Training Player 1 against Dealer
        policy_player1 = create_greedy_policy(q_dict_player1)
        _, exp_player1, _ = play_game(policy_player1, dealer_policy, deck, gamma=gamma)
        q_dict_player1 = update_Q(
            q_dict_player1, exp_player1, alpha=alpha, method=method
        )

        # Training Player 2 against Dealer
        policy_player2 = create_greedy_policy(q_dict_player2)
        _, exp_player2, _ = play_game(policy_player2, dealer_policy, deck, gamma=gamma)
        q_dict_player2 = update_Q(
            q_dict_player2, exp_player2, alpha=alpha, method=method
        )

    return q_dict_player1, q_dict_player2


def main(debug: bool = False, pretrain: bool = True, method: str = "qlearning"):
    deck = CardDeck()

    if debug:
        player1_report_callback = lambda txt: print(f"[bold blue]Player 1:[/] {txt}")
        player2_report_callback = lambda txt: print(f"[bold green]Player 2:[/] {txt}")
        game_report_callback = lambda txt: print(f"[bold red]Game:[/] {txt}")

        policy1 = random_policy
        policy2 = random_policy

        game_result, player1_experience, player2_experience = play_game(
            policy1,
            policy2,
            deck,
            game_report_callback=game_report_callback,
            player1_report_callback=player1_report_callback,
            player2_report_callback=player2_report_callback,
            debug=debug,
        )

        if game_result == 1:
            print("Player 1 wins!")
        elif game_result == -1:
            print("Player 2 wins!")
        else:
            print("The game is a draw.")

        print("Player 1 Experience:")
        for experience in player1_experience:
            print(experience)

        print("Player 2 Experience:")
        for experience in player2_experience:
            print(experience)

    q_dict_player1 = dict()
    q_dict_player2 = dict()

    q_dict_best_player1 = dict()
    q_dict_best_player2 = dict()

    if pretrain:
        q_dict_player1, q_dict_player2 = pre_train_players(
            deck, num_games=50000, method=method
        )
        pretrained_policy_player1 = create_greedy_policy(q_dict_best_player1)
        pretrained_policy_player2 = create_greedy_policy(q_dict_best_player2)

        print("Visualizing Pretrained Policy for Player 1:")
        visualize_policy(pretrained_policy_player1)
        print("Visualizing Pretrained Policy for Player 2:")
        visualize_policy(pretrained_policy_player2)

    res_best_player1 = -float("inf")
    res_best_player2 = -float("inf")

    for _ in trange(100000):
        # Create greedy policies for both players
        policy_player1 = create_greedy_policy(q_dict_player1)
        policy_player2 = create_greedy_policy(q_dict_player2)

        # Play games and update policies
        _, exp_player1, exp_player2 = play_game(
            policy_player1, policy_player2, deck, gamma=0.9
        )
        q_dict_player1 = update_Q(q_dict_player1, exp_player1, alpha=0.1, method=method)
        q_dict_player2 = update_Q(q_dict_player2, exp_player2, alpha=0.1, method=method)

        # Evaluate policies after each epoch
        res_player1, res_player2 = apply_policy_exhaustive(
            policy_player1, policy_player2, deck
        )

        # Update the best result and Q-dictionary for each player
        if res_player1.score > res_best_player1:
            q_dict_best_player1 = deepcopy(q_dict_player1)
            res_best_player1 = res_player1.score
        if res_player2.score > res_best_player2:
            q_dict_best_player2 = deepcopy(q_dict_player2)
            res_best_player2 = res_player2.score

    # Retrieve the best policies for final evaluation
    final_policy_player1 = create_greedy_policy(q_dict_best_player1)
    final_policy_player2 = create_greedy_policy(q_dict_best_player2)

    # Evaluate final policies for both players
    final_res_player1, final_res_player2 = apply_policy_exhaustive(
        final_policy_player1,
        final_policy_player2,
        deck,
        epoch_no=10000,
    )

    # Print final evaluation results
    print("Final Policy Evaluation Player 1:")
    print(final_res_player1)
    print("Final Policy Evaluation Player 2:")
    print(final_res_player2)

    # Visualize final policies
    print("Visualizing Final Policy for Player 1:")
    visualize_policy(final_policy_player1)
    print("Visualizing Final Policy for Player 2:")
    visualize_policy(final_policy_player2)


if __name__ == "__main__":
    main(debug=False, pretrain=True, method="sarsa")

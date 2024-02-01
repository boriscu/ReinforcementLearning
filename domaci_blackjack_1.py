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


def discounted_gains(rewards: list[float], gamma) -> list[float]:
    """
    Calculate the discounted gains (or returns) for each timestep in a sequence of rewards
    using a discount factor. The discount factor, gamma, is used to weigh rewards received
    in the future less than rewards received immediately, reflecting the principle that
    immediate rewards are more valuable than the same rewards received in the future.

    This function computes the discounted gain for each timestep in the sequence by
    summing all future rewards from that timestep, each discounted by how far in the
    future it is received. The gain at each timestep is calculated as:

    G_t = R_t+1 + gamma * R_t+2 + gamma^2 * R_t+3 + ... + gamma^(T-t-1) * R_T

    where G_t is the gain at time t, R_t is the reward received at time t, gamma is the
    discount factor (0 <= gamma <= 1), and T is the total number of timesteps.

    Args:
        rewards (list[float]): A sequence of rewards received over time, where each element
                               in the list represents the reward received at a particular timestep.
        gamma (float): The discount factor used to devalue future rewards relative to immediate
                       rewards. A gamma of 0 means "only value immediate rewards," while a
                       gamma of 1 means "value future rewards just as much as immediate rewards."

    Returns:
        list[float]: A list of discounted gains, where each element corresponds to the discounted
                     gain calculated from a particular timestep in the sequence of rewards.

    Example:
        If rewards = [1, 2, 3, 4] and gamma = 0.5, the function will calculate the discounted
        gain for each timestep, resulting in a list of gains, each calculated using the formula
        provided above.

    Note:
        - The length of the returned list of gains will be the same as the length of the input
          list of rewards.
        - This function is particularly useful in the context of reinforcement learning for
          calculating the expected returns from taking certain actions in specific states,
          taking into account the diminishing value of future rewards.
    """
    gains = [compute_gain(rewards[i:], gamma) for i in range(len(rewards))]
    return gains


def build_experience(
    player_log: TurnLog, opponent_log: TurnLog, result: int, gamma: float
) -> Experience:
    """
    Compute experience from the turn logs and the final result for a two-player game.

    Args:
        player_log (TurnLog): List of state-action pairs for the player.
        opponent_log (TurnLog): List of state-action pairs for the opponent.
        result (int): Final result of the game from the player's perspective (+1 win, 0 draw, -1 lose).
        gamma (float): Discount factor.

    Return:
        Experience: List of state-action-gain triples for the player.
    """
    player_rewards = [0 for _ in player_log]
    player_rewards[-1] = result

    player_gains = discounted_gains(player_rewards, gamma)
    player_experience = [
        (state, action, gain) for (state, action), gain in zip(player_log, player_gains)
    ]

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
    player1_starts: bool = True,
) -> tuple[int, Experience]:
    report = lambda txt: game_report_callback and game_report_callback(txt)

    # Initialize states for both players
    player1_state, player2_state = get_init_states(deck, debug)

    if player1_starts:
        # Player 1 takes their turn first
        player1_total, player1_log = play_turn(
            policy1, deck, player1_state, report_callback=player1_report_callback
        )
        # Then Player 2 takes their turn
        player2_total, player2_log = play_turn(
            policy2, deck, player2_state, report_callback=player2_report_callback
        )
    else:
        # Player 2 takes their turn first if player1_starts is False
        player2_total, player2_log = play_turn(
            policy2, deck, player2_state, report_callback=player2_report_callback
        )
        # Then Player 1 takes their turn
        player1_total, player1_log = play_turn(
            policy1, deck, player1_state, report_callback=player1_report_callback
        )

    # Determine the outcome of the game
    game_result = compare_totals(player1_total, player2_total)

    # Report the game outcome
    report(f"Game result: {game_result}")

    # Build experiences for both players
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
    """
    Conducts an exhaustive evaluation of two policies by simulating a series of games
    between them. It aggregates the outcomes of these games to provide a comprehensive
    assessment of each policy's performance.

    The function iterates through a specified number of epochs (games), where in each game,
    the two policies compete against each other. The outcomes of these games (win, loss, draw)
    are recorded and used to calculate the overall performance metrics for each policy.

    Parameters:
        policy1 (Policy): The first policy to be evaluated. A policy is a function that
                          decides an action based on the current game state.
        policy2 (Policy): The second policy to be evaluated.
        deck (CardDeck): An instance of a deck of cards, which will be used throughout the
                         game simulations to draw cards.
        epoch_no (int, optional): The number of games to simulate for the evaluation.
                                  Defaults to 5.
        gamma (float, optional): The discount factor used in the game's reward calculation.
                                 It's not directly used in this function but is passed to
                                 the game simulation function. Defaults to 1, indicating no
                                 discounting.

    Returns:
        tuple[PolicyTestingResult, PolicyTestingResult]: A tuple containing the testing
        results for both policies. Each `PolicyTestingResult` includes the number of games
        played, the total score, and the proportions of wins, draws, and losses.

    Each game result contributes to the scores and counts of wins, losses, and draws for
    both policies. The total score for a policy is adjusted by +1 for a win, -1 for a loss,
    and 0 for a draw. These metrics provide insights into the effectiveness and robustness
    of the evaluated policies under the simulation conditions set by the `deck` and the
    `gamma` factor.

    Note:
        - A positive score indicates a policy that tends to win more than lose, while a
          negative score indicates the opposite.
        - This function assumes that the policies are deterministic and does not account
          for stochastic effects in policy decisions. However, randomness in the deck
          shuffling and card drawing can still introduce variability in the game outcomes.
    """
    wins1, losses1, draws1, score1, games_no1 = 0, 0, 0, 0, 0
    wins2, losses2, draws2, score2, games_no2 = 0, 0, 0, 0, 0

    for epoch in range(epoch_no):
        player1_starts = (
            epoch % 2 == 0
        )  # Player 1 starts on even epochs, Player 2 on odd epochs

        res, exp_player1, exp_player2 = play_game(
            policy1, policy2, deck, gamma=gamma, player1_starts=player1_starts
        )

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
    """
    Create a dictionary mapping from game states to the gains associated with each possible action in those states, based on the given experiences.

    Each state in the game is represented by a tuple of three elements: the player's total card value, a boolean indicating whether the player has a usable ace, and the opponent's (or dealer's) visible card value. The gains dictionary maps these states to pairs of lists: one for the gains associated with choosing to HIT, and one for choosing to HOLD.

    The gains for each action are aggregated from the experiences collected during gameplay. An experience is a tuple consisting of a state, an action taken in that state (either HIT or HOLD), and the gain resulting from that action. The gain represents the immediate outcome or reward of taking the action, considering the future states until the end of the game.

    Interpretation of the Gains Dictionary:
    - Each key in the dictionary is a state, and the value is a pair of lists: the first list contains gains for HITTING, and the second list for HOLDING.
    - A negative gain value indicates that, on average, the action led to a decrease in the player's chance of winning from that state.
    - An empty list for an action means no experiences were recorded for that action in the given state, making its outcome unknown.

    Example:
    `GAINS DICT: {(15, True, 4): ([-0.81], []), (20, True, 4): ([-0.9], [])}`

    This example means:
    - For the state `(15, True, 4)`, the gain for choosing to HIT is `-0.81`, indicating an average negative outcome for this action. There's no data for choosing to HOLD, implying the outcome for this action is unknown in this state.
    - For the state `(20, True, 4)`, the gain for HITTING is `-0.9`, similarly indicating a negative outcome on average for this action. As with the previous state, there's no information about the outcome of HOLDING.

    Parameters:
    - experience (Experience): A list of tuples, where each tuple contains a state, an action, and a gain.

    Returns:
    - GainsDict: A dictionary where keys are states and values are tuples of two lists: the first list contains gains from HITTING in that state, and the second list contains gains from HOLDING.
    """
    gains_dict = dict()
    for state, action, gain in experience:
        action_gains = gains_dict.get(state, ([], []))
        action_gains[action.value].append(gain)
        gains_dict[state] = action_gains
    return gains_dict


# Incremental Monte Karlo
def update_q_value(current: float, target: float, alpha: float) -> float:
    """
    Update the given current Q-value estimate with a new target value using the incremental Monte Carlo update formula. This function applies the concept of temporal difference learning, specifically for updating action-value (Q-value) estimates in reinforcement learning.

    The formula for updating the Q-value is as follows:
    Q_new = Q_current + alpha * (target - Q_current)

    Where:
    - Q_current is the current estimate of the Q-value for a given state-action pair.
    - target is the observed return (or gain) from following the policy from the current state-action onward.
    - alpha is the learning rate, a parameter that determines the extent to which the newly acquired information overrides the old information. A value of 0 makes the agent not learn anything, while a value of 1 would make the agent consider only the most recent information.

    The update rule is a way to move the current Q-value estimate closer to the target value, with the learning rate controlling the size of the update step. This method is called incremental because it updates the estimates based on individual experiences or steps, rather than waiting for the completion of an entire episode.

    Args:
        current (float): The existing estimate of the Q-value for a certain state-action pair.
        target (float): The observed return or gain for taking an action in a given state and following the current policy thereafter.
        alpha (float): The learning rate, a value between 0 and 1 that controls how much the new information affects the current Q-value estimate.

    Returns:
        float: The updated Q-value estimate for the state-action pair.

    Note:
    - If either the current Q-value or the target value is `None`, the function returns the non-None value, assuming a default initialization in such cases. If both are `None`, the function returns `None`, indicating an error or unhandled case.
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


def update_Q(q_dict: QDict, experience: Experience, alpha=0.1):
    """
    Update the Q-value estimates for all state-action pairs encountered in a set of experiences.
    This function calculates the average gain (or return) from each state-action pair in the
    experience and updates the corresponding Q-value in the Q-dictionary using the incremental
    Monte Carlo update rule.

    The function operates in several steps:
    1. Convert the list of experiences into a gains dictionary that maps each state to the
    gains associated with HIT and HOLD actions.
    2. For each state in the gains dictionary, calculate the average gain for HIT and HOLD actions.
    3. Retrieve the current Q-values for HIT and HOLD actions for each state from the Q-dictionary.
    4. Update the Q-values based on the average gains using the formula:
    Q_new = Q_current + alpha * (gain - Q_current),
    where gain is the average gain from the experience for a given action, and alpha is the
    learning rate.
    5. Update the Q-dictionary with the new Q-values.

    Args:
        q_dict (QDict): A dictionary where keys are state tuples and values are tuples of
                        Q-values for HIT and HOLD actions, respectively. Each key-value pair
                        represents the Q-value estimates before updating.
        experience (Experience): A list of tuples, each containing a state, an action taken
                                in that state, and the gain (or return) obtained. The experience
                                represents the outcomes of actions taken in specific states
                                during gameplay.
        alpha (float): The learning rate, a parameter that determines the rate at which new
                    information is incorporated into the Q-value estimates. It ranges from
                    0 to 1, with higher values allowing faster learning at the risk of
                    instability.

    Returns:
        QDict: The updated Q-dictionary with revised Q-value estimates based on the provided
            experiences.

    Note:
        - This function assumes that gains for actions not taken in a given state are represented
        by an empty list in the gains dictionary, and it handles such cases by not updating
        the Q-value for those actions.
        - If the experience does not contain any instances of a state-action pair, the corresponding
        Q-value remains unchanged.
    """
    gains_dict = create_gains_dict(experience)

    for s, (gains_HIT, gains_HOLD) in gains_dict.items():
        target_value_HIT = np.mean(gains_HIT) if gains_HIT else None
        target_value_HOLD = np.mean(gains_HOLD) if gains_HOLD else None
        old_value_HIT, old_value_HOLD = q_dict.get(s, (None, None))
        q_value_HIT = update_q_value(old_value_HIT, target_value_HIT, alpha)
        q_value_HOLD = update_q_value(old_value_HOLD, target_value_HOLD, alpha)
        q_dict[s] = (q_value_HIT, q_value_HOLD)

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


def pre_train_players(
    deck: CardDeck, num_games: int, alpha: float = 0.1, gamma: float = 0.9
):
    """
    Pre-trains two players against a fixed dealer policy over a specified number of games to
    initialize or improve their Q-value dictionaries.

    Args:
        deck (CardDeck): An instance of the CardDeck class, which provides a deck of cards
                         for the game simulations.
        num_games (int): The number of games to play for pre-training. Each game contributes
                         to the update of the Q-value dictionaries for both players.
        alpha (float): The learning rate used in the Q-value update formula. It determines
                       how new information affects the existing Q-values. Defaults to 0.1.
        gamma (float): The discount factor used in calculating the future reward. It
                       represents the importance of future rewards. Defaults to 0.9.

    Returns:
        Tuple[Dict, Dict]: A tuple containing two dictionaries. Each dictionary maps state-action
                           pairs to Q-values for one of the two players. These dictionaries
                           represent the learned strategies based on the pre-training.

    The function iterates through the specified number of games, during which it:
    1. Utilizes the current Q-dictionary to create a greedy policy for each player.
    2. Simulates a game between the player (using the greedy policy) and the dealer (using a
       fixed policy).
    3. Updates the player's Q-dictionary based on the outcomes of the game using the Q-learning
       update rule.

    This pre-training process aims to give the players an initial strategy before they compete
    against each other or undergo further training. It's an essential step in preparing the
    agents for more complex interactions or for refining their strategies with additional
    training sessions.
    """
    q_dict_player1 = dict()
    q_dict_player2 = dict()

    for game_no in trange(num_games, desc="Pretraining"):
        player1_starts = game_no % 2 == 0
        # Training Player 1 against Dealer
        policy_player1 = create_greedy_policy(q_dict_player1)
        _, exp_player1, _ = play_game(
            policy_player1,
            dealer_policy,
            deck,
            gamma=gamma,
            player1_starts=player1_starts,
        )
        q_dict_player1 = update_Q(q_dict_player1, exp_player1, alpha=alpha)

        # Training Player 2 against Dealer
        policy_player2 = create_greedy_policy(q_dict_player2)
        _, exp_player2, _ = play_game(
            policy_player2,
            dealer_policy,
            deck,
            gamma=gamma,
            player1_starts=player1_starts,
        )
        q_dict_player2 = update_Q(q_dict_player2, exp_player2, alpha=alpha)

    return q_dict_player1, q_dict_player2


def main(
    debug: bool = False,
    pretrain: bool = True,
    epochs: int = 10000,
    epochs_pretrain: int = 50000,
):
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
            deck, num_games=epochs_pretrain
        )
        pretrained_policy_player1 = create_greedy_policy(q_dict_best_player1)
        pretrained_policy_player2 = create_greedy_policy(q_dict_best_player2)

        print("Visualizing Pretrained Policy for Player 1:")
        visualize_policy(pretrained_policy_player1)
        print("Visualizing Pretrained Policy for Player 2:")
        visualize_policy(pretrained_policy_player2)

    res_best_player1 = -float("inf")
    res_best_player2 = -float("inf")

    for epoch in trange(epochs):
        player_1_starts = epoch % 2 == 0
        # Create greedy policies for both players
        policy_player1 = create_greedy_policy(q_dict_player1)
        policy_player2 = create_greedy_policy(q_dict_player2)

        # Play games and update policies
        _, exp_player1, exp_player2 = play_game(
            policy_player1,
            policy_player2,
            deck,
            gamma=0.9,
            player1_starts=player_1_starts,
        )
        q_dict_player1 = update_Q(q_dict_player1, exp_player1, alpha=0.1)
        q_dict_player2 = update_Q(q_dict_player2, exp_player2, alpha=0.1)

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
        epoch_no=10000,  # More comprehensive evaluation
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


# Execute the main function
if __name__ == "__main__":
    main(debug=False, pretrain=True, epochs=100000, epochs_pretrain=50000)

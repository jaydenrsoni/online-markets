import numpy as np
import math


# in p3 -> n rounds, k actions, k x k matrix of action pair payoffs
def double_exponential_weights(n, k, h, lr, mat, verbose=True):

    action_payoffs1 = [0] * k  # V vector holding cumulative payoffs of each action
    action_payoffs2 = [0] * k  # V vector holding cumulative payoffs of each action
    actions1 = []  # list of actions algorithm takes
    actions2 = []  # list of actions algorithm takes
    total_payoff1 = 0  # payoff of actions algorithm takes
    total_payoff2 = 0  # payoff of actions algorithm takes

    # simulate n rounds
    for _ in range(n):

        # player 1
        raw_probabilities = [math.pow((1 + lr), action_payoffs1[j]/h) for j in range(k)]
        norm_probabilities = np.divide(raw_probabilities, math.fsum(raw_probabilities))
        action1 = np.random.choice(k, 1, p=norm_probabilities)[0]
        if verbose:
            print("player 1 probabilities: ", norm_probabilities)
        
        actions1.append(action1)

        # player 2
        raw_probabilities = [math.pow((1 + lr), action_payoffs2[j]/h) for j in range(k)]
        norm_probabilities = np.divide(raw_probabilities, math.fsum(raw_probabilities))
        action2 = np.random.choice(k, 1, p=norm_probabilities)[0]
        if verbose:
            print("player 2 probabilities: ", norm_probabilities)

        actions2.append(action2)

        # update player payoffs
        payoff1, payoff2 = mat[action1][action2]
        total_payoff1 += payoff1
        total_payoff2 += payoff2

        # update V with payoffs from current round
        action_payoffs1 = np.add(action_payoffs1, [mat[i][action2][0] for i in range(k)])
        action_payoffs2 = np.add(action_payoffs2, [mat[action1][i][1] for i in range(k)])
        if verbose:
            print("action payoffs1: ", action_payoffs1)
            print("action payoffs2: ", action_payoffs2)

    # Player 1
    print("Player1: ")
    print("actions: ", actions1)
    print("total payoff: ", total_payoff1)

    # Player 2
    print("Player2: ")
    print("actions: ", actions2)
    print("total payoff: ", total_payoff2)

    return None


def double_ftpl(n, k, h, lr, mat, verbose=True):

    action_payoffs1 = [0] * k  # V vector holding cumulative payoffs of each action
    action_payoffs2 = [0] * k  # V vector holding cumulative payoffs of each action
    actions1 = []  # list of actions algorithm takes
    actions2 = []  # list of actions algorithm takes
    total_payoff1 = 0  # payoff of actions algorithm takes
    total_payoff2 = 0  # payoff of actions algorithm takes

    hallucinations1 = np.random.geometric(p=lr, size=k) * h  # only used for ftpl
    hallucinations2 = np.random.geometric(p=lr, size=k) * h  # only used for ftpl

    # simulate n rounds
    for _ in range(n):

        # player 1
        action1 = np.argmax(np.add(hallucinations1, action_payoffs1))
        actions1.append(action1)

        # player 2
        action2 = np.argmax(np.add(hallucinations2, action_payoffs2))
        actions2.append(action2)

        # update player payoffs
        payoff1, payoff2 = mat[action1][action2]
        total_payoff1 += payoff1
        total_payoff2 += payoff2

        # update V with payoffs from current round
        action_payoffs1 = np.add(action_payoffs1, [mat[i][action2][0] for i in range(k)])
        action_payoffs2 = np.add(action_payoffs2, [mat[action1][i][1] for i in range(k)])
        if verbose:
            print("action payoffs1: ", action_payoffs1)
            print("action payoffs2: ", action_payoffs2)

    # Player 1
    print("Player1: ")
    print("actions: ", actions1)
    print("total payoff: ", total_payoff1)

    # Player 2
    print("Player2: ")
    print("actions: ", actions2)
    print("total payoff: ", total_payoff2)

    return None


def double_ftpl_nonuniform_bounds(n, k, lr, mat, verbose=True):

    action_payoffs1 = [0] * k  # V vector holding cumulative payoffs of each action
    action_payoffs2 = [0] * k  # V vector holding cumulative payoffs of each action
    actions1 = []  # list of actions algorithm takes
    actions2 = []  # list of actions algorithm takes
    total_payoff1 = 0  # payoff of actions algorithm takes
    total_payoff2 = 0  # payoff of actions algorithm takes

    hallucinations1 = np.multiply(np.random.geometric(p=lr, size=k),
                                  [max([bots[i][j][0] for j in range(k)]) for i in range(k)])
    hallucinations2 = np.multiply(np.random.geometric(p=lr, size=k),
                                  [max([bots[j][i][1] for j in range(k)]) for i in range(k)])

    # simulate n rounds
    for _ in range(n):

        # player 1
        action1 = np.argmax(np.add(hallucinations1, action_payoffs1))
        actions1.append(action1)

        # player 2
        action2 = np.argmax(np.add(hallucinations2, action_payoffs2))
        actions2.append(action2)

        # update player payoffs
        payoff1, payoff2 = mat[action1][action2]
        total_payoff1 += payoff1
        total_payoff2 += payoff2

        # update V with payoffs from current round
        action_payoffs1 = np.add(action_payoffs1, [mat[i][action2][0] for i in range(k)])
        action_payoffs2 = np.add(action_payoffs2, [mat[action1][i][1] for i in range(k)])
        if verbose:
            print("action payoffs1: ", action_payoffs1)
            print("action payoffs2: ", action_payoffs2)

    # Player 1
    print("Player1: ")
    print("actions: ", actions1)
    print("total payoff: ", total_payoff1)

    # Player 2
    print("Player2: ")
    print("actions: ", actions2)
    print("total payoff: ", total_payoff2)

    return None


def theoretical_lr(n, k):
    epsilon = math.sqrt(math.log(k) / n)
    print("\nTheoretically optimal learning rate: ", round(epsilon, 5))
    return epsilon


if __name__ == "__main__":

    # payoff matrices indexed as follows: bots[row][col][player]

    # (10, 5)   (0, 0)
    # (0, 0)    (5, 10)
    bots = (((10, 5), (0, 0)), ((0, 0), (5, 10)))

    n1 = 1000
    k1 = 2
    h1 = 10
    lr1 = theoretical_lr(n1, k1)

    print("\nBoth Exponential Weights:\n")
    for _ in range(3):
        double_exponential_weights(n1, k1, h1, lr1, bots, False)
    print("\nBoth Follow the Perturbed Leader:\n")
    for _ in range(3):
        double_ftpl(n1, k1, h1, lr1, bots, False)
    print("\nBoth Follow the Perturbed Leader with non-uniform bounds:\n")
    for _ in range(3):
        double_ftpl_nonuniform_bounds(n1, k1, lr1, bots, False)

import numpy as np
import math

##  (10, 5)   (0, 0)
##  (0, 0)    (5, 10)
bots = (((10,5), (0,0)), ((0,0), (5,10)))

# in p3 - n rounds, k actions, payoffs depend on opponent strategy
def double_exponential_weights(lr, h, n, verbose=True):
    
    k = 2 # num actions

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
        payoff1, payoff2 = bots[action1][action2]
        total_payoff1 += payoff1
        total_payoff2 += payoff2

        # update V with payoffs from current round
        # TODO: handle more than 2 actions
        action_payoffs1 = np.add(action_payoffs1, (bots[0][action2][0], bots[1][action2][0]))
        action_payoffs2 = np.add(action_payoffs2, (bots[action1][0][1], bots[action1][1][1]))
        if verbose:
            print("action payoffs1: ", action_payoffs1)
            print("action payoffs2: ", action_payoffs2)

    # # calculate OPT and regret
    # best_in_hindsight_payoff = max(action_payoffs)
    # regret = (best_in_hindsight_payoff - total_payoff) / len(data)

    # Player 1
    print("Player1: ")
    print("actions: ", actions1)
    print("total payoff: ", total_payoff1)

    # Player 2
    print("Player2: ")
    print("actions: ", actions2)
    print("total payoff: ", total_payoff2)

    return None


def theoretical_lr(n, k, h):
    epsilon = math.sqrt(math.log(k) / n)
    print("\nTheoretically optimal learning rate: ", epsilon)
    return epsilon


if __name__ == "__main__":
    print(bots[0])          # row 0
    print(bots[0][0])       # row 0, col 0
    print(bots[0][0][0])    # row 0, col 0, player 0

    n = 1000
    k = 2
    h = 10
    double_exponential_weights(theoretical_lr(n, k, h), h, n, False)
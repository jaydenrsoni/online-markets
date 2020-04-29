import numpy as np
import math
import os, sys, csv

# payoff data formatting examples
# 2 actions, 3 rounds [[0, 0], [0, 0], [0, 0]]
# 3 actions, 2 rounds [[0, 0, 0], [0, 0, 0]]


# bernoulli payoffs with probability p for each action k in any round n
def gen_data1(n, k, p, verbose=True):
    payoffs = [[0] * k for _ in range(n)]
    for i in range(n):
        for j in range(k):
            if np.random.random_sample() < p[j]:
                payoffs[i][j] = 1
    if verbose:
        print(payoffs)
    return payoffs


# bernoulli variable turns binomial payouts on/off with probability p1 for each action k in any round n
# each action k has unique binomial variable with particular number of trials (kns) and probability of success (p2)
def gen_data2(n, k, p1, kns, p2, verbose=True):
    payoffs = [[0] * k for _ in range(n)]
    for i in range(n):
        for j in range(k):
            if np.random.random_sample() < p1[j]:
                payoffs[i][j] = np.random.binomial(kns[j], p2[j])
    if verbose:
        print(payoffs)
    return payoffs


def online_learning(data, lr, h, algo="ew", verbose=True):
    k = len(data[0])  # num actions
    hallucinations = np.random.geometric(p=lr, size=k) * h  # only used for ftpl
    if algo == "ftpl":
        print("hallucinations: ", hallucinations)

    action_payoffs = [0] * k  # V vector holding cumulative payoffs of each action
    actions = []  # list of actions algorithm takes
    total_payoff = 0  # payoff of actions algorithm takes

    # loop through payoffs round-by-round
    for curr_payoffs in data:

        # choose an action action using given algorithm
        if algo == "ew":
            try:
                raw_probabilities = [math.pow((1 + lr), action_payoffs[j]/h) for j in range(k)]
            except OverflowError:
                print("issue")
            norm_probabilities = np.divide(raw_probabilities, math.fsum(raw_probabilities))
            action = np.random.choice(k, 1, p=norm_probabilities)[0]
            if verbose:
                print("probabilities: ", norm_probabilities)

        elif algo == "ftpl":
            action = np.argmax(np.add(hallucinations, action_payoffs))

        else:
            raise Exception("not a valid algorithm")

        actions.append(action)
        total_payoff += curr_payoffs[action]

        # update V with payoffs from current round
        action_payoffs = np.add(action_payoffs, curr_payoffs)
        if verbose:
            print("action payoffs: ", action_payoffs)

    # calculate OPT and regret
    best_in_hindsight_payoff = max(action_payoffs)
    regret = (best_in_hindsight_payoff - total_payoff) / len(data)

    print("actions (max of last 25 shown): ", actions[len(actions) - 20:])
    print("total payoff: ", total_payoff)
    print("regret: ", regret)
    return total_payoff, best_in_hindsight_payoff


def theoretical_lr(n, k, data):
    epsilon = math.sqrt(math.log(k) / n)
    print("\nTheoretically optimal learning rate: ", epsilon)

    print("\nExponential Weights (theo): ")
    online_learning(data, epsilon, 1, algo="ew", verbose=False)
    print("\nFollow the Perturbed Leader (theo): ")
    online_learning(data, epsilon, 1, algo="ftpl", verbose=False)


def empirical_lr(data):
    max_idx_ew = -1
    max_val_ew = 0
    max_idx_ftpl = -1
    max_val_ftpl = 0
    possible_epsilon = [x / 1000 for x in range(500)]  # 0.001 to 0.499 interval of 0.001
    possible_epsilon.remove(0)
    sys.stdout = open(os.devnull, 'w')  # stops printing
    for idx, lr in enumerate(possible_epsilon):  # this loop takes a little while
        val_ew, opt = online_learning(data, lr, 1, algo="ew", verbose=False)
        val_ftpl, opt = online_learning(data, lr, 1, algo="ftpl", verbose=False)
        if val_ew > max_val_ew:
            max_val_ew = val_ew
            max_idx_ew = idx
        if val_ftpl > max_val_ftpl:
            max_val_ftpl = val_ftpl
            max_idx_ftpl = idx
    sys.stdout = sys.__stdout__  # enables printing again
    print("\nEmprically optimal learning rate for EW:", possible_epsilon[max_idx_ew])
    print("total payoff: ", max_val_ew)
    print("regret: ", (opt - max_val_ew) / len(data))
    print("\nEmprically optimal learning rate for FTPL:", possible_epsilon[max_idx_ftpl])
    print("total payoff: ", max_val_ftpl)
    print("regret: ", (opt - max_val_ftpl) / len(data))


def load_bids():
    with open('bid_data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        ct = 0
        global values, bids
        values = []
        bids = []
        for row in csv_reader:
            if ct == 0:
                ct += 1
                continue
            values.append(float(row[0]))
            if not row[1] == " ":
                bids.append(float(row[1]))
    return bids


def first_price_auction_payoffs(value, num_discrete_actions, num_rounds, verbose=True):
    bid_data = load_bids()
    n = len(bid_data)

    if num_rounds > n:
        indices = np.random.choice(n, num_rounds, replace=True)
        bid_data = [bid_data[i] for i in indices]
        n = len(bid_data)

    bid_space = [x for x in np.linspace(0, value, num_discrete_actions)]  # all integers between 0 to value, inclusive
    k = len(bid_space)

    payoffs = [[0] * k for _ in range(n)]
    for i in range(n):
        for j in range(k):
            if bid_space[j] > bid_data[i]:
                payoffs[i][j] = value - bid_space[j]
    if verbose:
        print(payoffs)

    # calculate theoretically optimal learning rate
    theo_lr = math.sqrt(math.log(k)/n)

    return payoffs, theo_lr


if __name__ == '__main__':

    np.random.seed(100)

    # Part 1
    print("\nStarting Part 1...")

    # parameter set
    n = 1000
    k = 10
    p = [0.1 * i + 0.05 for i in range(10)]
    data1 = gen_data1(n, k, p, verbose=False)

    n2 = 1000
    k2 = 4
    p1 = [0.2, 0.25, 0.3, 0.35]
    kns = [10] * 4
    p2 = [0.5, 0.45, 0.4, 0.35]
    data2 = gen_data2(n2, k2, p1, kns, p2, verbose=False)

    # run learning algorithms with optimal learning rates
    theoretical_lr(n, k, data1)
    empirical_lr(data1)
    theoretical_lr(n2, k2, data2)
    empirical_lr(data2)

    # Part 2
    print("\n\nStarting Part 2...")
    val1 = 20.7
    data2_1, eps2 = first_price_auction_payoffs(val1, 21, 200, verbose=False)
    print("\nExponential Weights (20.7): ")
    online_learning(data2_1, eps2, val1, algo="ew", verbose=False)
    print("\nFollow the Perturbed Leader (20.7): ")
    online_learning(data2_1, eps2, val1, algo="ftpl", verbose=False)
    # TODO: repeat for other vals, empirical lr for part 2, optimize discretization/lr to maximize payoff

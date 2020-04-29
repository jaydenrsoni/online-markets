import numpy as np
import math
import os
import sys
import csv

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
            raw_probabilities = [math.pow((1 + lr), action_payoffs[j]/h) for j in range(k)]
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
    print("\nEmpirically optimal learning rate for EW:", possible_epsilon[max_idx_ew])
    print("total payoff: ", max_val_ew)
    print("regret: ", (opt - max_val_ew) / len(data))
    print("\nEmpirically optimal learning rate for FTPL:", possible_epsilon[max_idx_ftpl])
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


def first_price_auction_results(value, num_rounds, num_discrete_actions, emp_lr):
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

    # calculate theoretically optimal learning rate
    theo_lr = math.sqrt(math.log(k)/n)

    theo_ew = online_learning(payoffs, theo_lr, value, algo="ew", verbose=False)[0]
    emp_ew = online_learning(payoffs, emp_lr, value, algo="ew", verbose=False)[0]
    theo_ftpl = online_learning(payoffs, theo_lr, value, algo="ftpl", verbose=False)[0]
    emp_ftpl = online_learning(payoffs, emp_lr, value, algo="ftpl", verbose=False)[0]

    return theo_ew, emp_ew, theo_ftpl, emp_ftpl, theo_lr


def first_price_auction(value, num_rounds, opt, verbose=True):
    print("\nRunning a first price auction with value = ", value)
    possible_epsilon = [x / 100 for x in range(50)]  # 0.01 to 0.49, interval of 0.01
    num_action_bins = [x for x in range(2, int(value*2), 2)]  # bin size from value/2 down to ~ 0.5, step size of 2

    max_theo_ew = (0, 0, 0)  # payoff, theo_lr, num_action_bins
    max_emp_ew = (0, 0, 0)   # payoff, emp_lr, num_action_bins
    max_theo_ftpl = (0, 0, 0)
    max_emp_ftpl = (0, 0, 0)

    sys.stdout = open(os.devnull, 'w')  # stops printing
    for lr in possible_epsilon:
        for k in num_action_bins:
            theo_ew, emp_ew, theo_ftpl, emp_ftpl, theo_lr = first_price_auction_results(value, num_rounds, k, lr)
            if theo_ew > max_theo_ew[0]:
                max_theo_ew = (theo_ew, theo_lr, k)
            if emp_ew > max_emp_ew[0]:
                max_emp_ew = (emp_ew, lr, k)
            if theo_ftpl > max_theo_ftpl[0]:
                max_theo_ftpl = (theo_ftpl, theo_lr, k)
            if emp_ftpl > max_emp_ftpl[0]:
                max_emp_ftpl = (emp_ftpl, lr, k)

    best = max([max_theo_ew, max_emp_ew, max_theo_ftpl, max_emp_ftpl], key=lambda x: x[0])
    regret = opt - best[0]/num_rounds

    sys.stdout = sys.__stdout__  # enables printing again
    if verbose:
        print("\nAlgorithm Payoffs: ", [x[0] for x in [max_theo_ew, max_emp_ew, max_theo_ftpl, max_emp_ftpl]])
        print("\nOptimal Tuning...")
        print("learning rate: ", best[1])
        print("discrete actions: ", best[2])
        print("payoff: ", best[0])
        print("regret: ", regret)

    return best, regret


if __name__ == '__main__':

    np.random.seed(100)

    # Part 1
    print("\nStarting Part 1...")

    # bernoulli payoff matrix generation
    data1a = gen_data1(1000, 10, [0.1 * i + 0.05 for i in range(10)], verbose=False)
    data1b = gen_data1(100, 10, [0.1 * i + 0.05 for i in range(10)], verbose=False)
    data1c = gen_data1(500, 2, [0.4, 0.6], verbose=False) + gen_data1(500, 2, [0.6, 0.4], verbose=False)
    data1d = gen_data1(500, 5, [0.5] * 5, verbose=False)

    # payoff matrix generation with gen_data2(num rounds, num actions, sale ps, binomial amt trials, binomial amt ps)
    data2a = gen_data2(1000, 4, [0.2, 0.25, 0.3, 0.35], [10] * 4, [0.5, 0.45, 0.4, 0.35], verbose=False)
    data2b = gen_data2(1000, 4, [0.2, 0.25, 0.3, 0.35], [10, 9, 8, 7], [0.5] * 4, verbose=False)
    data2c = gen_data2(1000, 4, [0.2] * 4, [2, 4, 6, 8], [0.8, 0.6, 0.4, 0.2], verbose=False)
    data2d = gen_data2(100, 4, [0.2] * 4, [10] * 4, [0.5, 0.45, 0.4, 0.35], verbose=False)

    # run learning algorithms with optimal learning rates
    theoretical_lr(1000, 10, data1a)
    empirical_lr(data1a)
    theoretical_lr(100, 10, data1b)
    empirical_lr(data1b)
    theoretical_lr(500, 2, data1c)
    empirical_lr(data1c)
    theoretical_lr(500, 5, data1d)
    empirical_lr(data1d)

    theoretical_lr(1000, 4, data2a)
    empirical_lr(data2a)
    theoretical_lr(1000, 4, data2b)
    empirical_lr(data2b)
    theoretical_lr(1000, 4, data2c)
    empirical_lr(data2c)
    theoretical_lr(1000, 4, data2d)
    empirical_lr(data2d)

    # Part 2
    print("\n\nStarting Part 2...")
    first_price_auction(20.7, 300, 1.47)
    first_price_auction(38.2, 300, 5.1)
    first_price_auction(51.6, 300, 9.07)
    first_price_auction(93.7, 300, 30.31)

    for n in range(100, 1001, 100):
        print(first_price_auction(51.6, n, 9.07, verbose=False))

import numpy as np
import math
import os, sys, csv

def get_bid():
    with open('bid_data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        ct = 0
        global values, bids
        values = []
        bids = []
        for row in csv_reader:
            if ct == 0:
                ct+= 1
                continue
            values.append(row[0])
            bids.append(row[1])
            
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


def gen_data2():
    return 0    


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
    return total_payoff


if __name__ == '__main__':

    # basic testing
    test_data = gen_data1(10, 3, [0.5] * 3)
    online_learning(test_data, 0.5, 1, "ew")
    online_learning(test_data, 0.5, 1, "ftpl")

    # parameter set
    n = 1000
    k = 10
    p = [0.5] * 10
    data1 = gen_data1(n, k, p, verbose=False)

    # theoretically optimal learning rate
    epsilon = math.sqrt(math.log(k)/n)
    print("\nLearning rate: ", epsilon)

    print("\nExponential Weights (theo): ")
    online_learning(data1, epsilon, 1, algo="ew", verbose=False)
    print("\nFollow the Perturbed Leader (theo): ")
    online_learning(data1, epsilon, 1, algo="ftpl", verbose=False)

    find empirically optimal learning rate
    max_idx_ew = -1
    max_val_ew = 0
    max_idx_ftpl = -1
    max_val_ftpl = 0
    possible_epsilon = [x/1000 for x in range(300)] #0.001 to 0.299 interval of 0.001
    possible_epsilon.remove(0)
    sys.stdout = open(os.devnull, 'w') #stops printing
    for idx,lr in enumerate(possible_epsilon): #this loop takes a little while
        val_ew =online_learning(data1, lr, 1, algo="ew", verbose=False)
        val_ftpl = online_learning(data1, lr, 1, algo="ftpl", verbose=False)
        if val_ew > max_val_ew:
            max_val_ew = val_ew
            max_idx_ew = idx
        if val_ftpl > max_val_ftpl:
            max_val_ftpl = val_ftpl
            max_idx_ftpl = idx
    sys.stdout = sys.__stdout__ #enables printing again
    print("\nEmprically optimal learning rate for EW:", possible_epsilon[max_idx_ew])
    print("payoff: ", max_val_ew)
    print("\nEmprically optimal learning rate for FTPL:", possible_epsilon[max_idx_ftpl])
    print("payoff: ", max_val_ftpl)

    #part 2
    get_bid()
    
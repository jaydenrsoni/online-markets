import numpy as np
import math

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

    # empirically optimal learning rate

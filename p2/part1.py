import numpy as np
import math

def dataset1(n, k, p):
    ## 2 actions, 3 rounds [[0, 0], [0, 0], [0, 0]]
    payoffs = [[0] * k for _ in range(n)]
    for i in range(n):
        for j in range(k):
            if np.random.random_sample() < p[j]:
                payoffs[i][j] = 1
    print(payoffs)
    return payoffs


def dataset2():
    return 0    


def exponential_weights(data, lr, h):
    action_payoffs = [0] * len(data[0])
    actions = []
    total_payoff = 0

    for round in data:
        probabilities = []
        sum_probabilities = 0
        for i in range(len(round)):
            p = math.pow((1 + lr), action_payoffs[i]/h)
            probabilities.append(p)
            sum_probabilities += p
        probabilities = [p/sum_probabilities for p in probabilities]
        print(probabilities)

        action = np.random.choice(len(round), 1, p=probabilities)[0]
        actions.append(action)
        total_payoff += round[action]

        # update V
        for i in range(len(round)):
            action_payoffs[i] += round[i]
        print("action payoffs: ", action_payoffs)
    print(total_payoff, actions)

    best_in_hindsight_payoff = max(action_payoffs)
    regret = (best_in_hindsight_payoff - total_payoff)/len(data)
    print(regret)


def follow_the_perturbed_leader(data, lr, h):
    k = len(data[0])  # num actions
    hallucinations = np.random.geometric(p=lr, size=k) * h
    print(hallucinations)

    action_payoffs = [0] * len(data[0])
    actions = []
    total_payoff = 0

    for round in data:
        action = np.argmax(np.add(hallucinations, action_payoffs))
        actions.append(action)
        total_payoff += round[action]

        # update V
        for i in range(len(round)):
            action_payoffs[i] += round[i]
        print("action payoffs: ", action_payoffs)
    print(total_payoff, actions)

    best_in_hindsight_payoff = max(action_payoffs)
    regret = (best_in_hindsight_payoff - total_payoff) / len(data)
    print(regret)


def online_learning(data, lr, h, type="ew"):



if __name__ == '__main__':
    data1 = dataset1(10, 3, [0.5] * 3)
    exponential_weights(data1, 0.5, 1)
    follow_the_perturbed_leader(data1, 0.5, 2)
    # # theoretically optimal learning rate
    # exponential_weights(dataset1(), 0)
    # follow_the_perturbed_leader(dataset1(), 0)

    # # empirically optimal learning rate
    # exponential_weights(dataset1(), 0)
    # follow_the_perturbed_leader(dataset1(), 0)

    # # theoretically optimal learning rate
    # exponential_weights(dataset2(), 0)
    # follow_the_perturbed_leader(dataset2(), 0)

    # # empirically optimal learning rate
    # exponential_weights(dataset2(), 0)
    # follow_the_perturbed_leader(dataset2(), 0)

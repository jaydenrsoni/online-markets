import numpy as np
import matplotlib.pyplot as plt
import math


# num_bidders bidders with values on distribution f, num_items items for n rounds and k possible reserve prices [0,1)
def gen_data(n, k, num_bidders, num_items, f, verbose=False):
    payoffs = [[0] * k for _ in range(n)]
    player_bids = [[f() for _ in range(num_bidders)] for _ in range(n)]
    for i in range(n):
        for j in range(k):
            bids = set(player_bids[i])
            max_bids = []
            for _ in range(num_items):
                max_bids.append(max(bids))
                bids = bids - {max(bids)}
            price = max(bids)
            for bid in max_bids:
                payoffs[i][j] += max(price, j/k) if bid >= j/k else 0
    if verbose:
        print(player_bids[0])
        print(payoffs[0])
    return payoffs


def quadratic_dist():
    return math.sqrt(np.random.random_sample())


# part 1
def online_reserve_pricing(data, lr, algo="ew", verbose=False):
    n = len(data)                    # num rounds
    k = len(data[0])                 # num actions
    h = 1

    hallucinations = np.random.geometric(p=lr, size=k) * h  # only used for ftpl
    if verbose and algo == "ftpl":
        print("hallucinations: ", hallucinations)

    action_payoffs = [0] * k  # V vector holding cumulative payoffs of each action
    actions = []  # list of actions algorithm takes
    payoffs = []  # list of payoffs algorithm achieves

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

        actions.append(action/k)
        payoffs.append(curr_payoffs[action])

        # update V with payoffs from current round
        action_payoffs = np.add(action_payoffs, curr_payoffs)
        if verbose:
            print("action payoffs: ", action_payoffs)

    # calculate OPT and regret
    best_in_hindsight_payoff = max(action_payoffs)
    regret = (best_in_hindsight_payoff - sum(payoffs)) / len(data)

    print("last actions: ", actions[n - 100:])
    print("last payoffs: ", payoffs[n - 100:])
    print("converged action: ", sum(actions[n - 100:])/100)
    print("converged payoff: ", sum(payoffs[n - 100:])/100)
    print("best in hindsight action: ", np.argmax(action_payoffs)/k)
    print("best in hindsight payoff: ", best_in_hindsight_payoff/n)
    print("average payoff: ", sum(payoffs)/n)
    print("regret: ", regret, "\n")
    return actions


def plot_results(data, lr, title):
    actions1 = online_reserve_pricing(data, lr, algo="ew")
    actions2 = online_reserve_pricing(data, lr, algo="ftpl")
    plt.plot(range(n1), actions1, label="ew")
    plt.plot(range(n1), actions2, label="ftpl")
    plt.xlabel("Round")
    plt.ylabel("Reserve Price")
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    print("\nPart 1\n")

    n1 = 10000
    k1 = 100
    theo_lr = math.sqrt(math.log(k1) / n1)

    # uniform [0,1]
    data1 = gen_data(n1, k1, 2, 1, np.random.random_sample)
    data2 = gen_data(n1, k1, 5, 1, np.random.random_sample)
    data3 = gen_data(n1, k1, 10, 1, np.random.random_sample)
    data4 = gen_data(n1, k1, 10, 4, np.random.random_sample)

    plot_results(data1, theo_lr, "uni 2 bidders, 1 item, lr = " + str(theo_lr))
    plot_results(data1, 0.15, "uni 2 bidders, 1 item, lr = 0.15")
    plot_results(data2, theo_lr, "uni 5 bidders, 1 item, lr = " + str(theo_lr))
    plot_results(data2, 0.1, "uni 5 bidders, 1 item, lr = 0.1")
    plot_results(data3, theo_lr, "uni 10 bidders, 1 item, lr = " + str(theo_lr))
    plot_results(data4, theo_lr, "uni 10 bidders, 4 items, lr = " + str(theo_lr))

    # quadratic [0,1]
    data5 = gen_data(n1, k1, 2, 1, quadratic_dist)
    data6 = gen_data(n1, k1, 5, 1, quadratic_dist)
    data7 = gen_data(n1, k1, 10, 1, quadratic_dist)
    data8 = gen_data(n1, k1, 10, 6, quadratic_dist)

    plot_results(data5, theo_lr, "quad 2 bidders, 1 item, lr = " + str(theo_lr))
    plot_results(data5, 0.1, "quad 2 bidders, 1 item, lr = 0.1")
    plot_results(data6, theo_lr, "quad 5 bidders, 1 item, lr = " + str(theo_lr))
    plot_results(data6, 0.06, "quad 5 bidders, 1 item, lr = 0.06")
    plot_results(data7, theo_lr, "quad 10 bidders, 1 item, lr = " + str(theo_lr))
    plot_results(data8, theo_lr, "quad 10 bidders, 4 items, lr = " + str(theo_lr))

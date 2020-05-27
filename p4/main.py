import numpy as np
import matplotlib.pyplot as plt
import math


def online_reserve_pricing(data, lr, h, algo="ew", verbose=False):
    n = len(data)                    # num rounds
    k = len(data[0])                 # num actions

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

        actions.append(round(action/k * h, 2))
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
    print("best in hindsight action: ", round(np.argmax(action_payoffs)/k * h, 2))
    print("best in hindsight payoff: ", best_in_hindsight_payoff/n)
    print("average payoff: ", sum(payoffs)/n)
    print("regret: ", regret, "\n")
    print(action_payoffs[int(k/2)]/n)
    return actions


def plot_results(data, lr, title, h=1):
    actions1 = online_reserve_pricing(data, lr, h, algo="ew")
    actions2 = online_reserve_pricing(data, lr, h, algo="ftpl")
    plt.plot(range(n1), actions1, label="ew")
    plt.plot(range(n1), actions2, label="ftpl")
    plt.xlabel("Round")
    plt.ylabel("Reserve Price")
    plt.title(title)
    plt.show()


def quadratic_dist():
    return math.sqrt(np.random.random_sample())


# num_bidders bidders with values from distribution f [0,1), num_items items for n rounds and k possible reserve prices
def gen_data(n, k, num_bidders, num_items, f, h=1, verbose=False):
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
                payoffs[i][j] += max(price, j/k * h) if bid >= j/k * h else 0
    if verbose:
        print(player_bids[0])
        print(payoffs[0])
    return payoffs


# revenue for selling intro between num_part participants with values from distributions dists [0,h], n rounds and k
# possible v1 + v2 + ... thresholds [0,h * num_bidders]
def gen_data2(n, k, num_part, dists, h=1, verbose=False):
    payoffs = [[0] * k for _ in range(n)]
    for i in range(n):
        values = [dists[j]() for j in range(num_part)]
        for j in range(k):
            thresh = (j/k) * h * num_part
            payoffs[i][j] += thresh * num_part - math.fsum(values)*(num_part-1) if sum(values) >= thresh else 0
    if verbose:
        print(payoffs[0])
    return payoffs


def part1():
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
    data8 = gen_data(n1, k1, 10, 4, quadratic_dist)

    plot_results(data5, theo_lr, "quad 2 bidders, 1 item, lr = " + str(theo_lr))
    plot_results(data5, 0.1, "quad 2 bidders, 1 item, lr = 0.1")
    plot_results(data6, theo_lr, "quad 5 bidders, 1 item, lr = " + str(theo_lr))
    plot_results(data6, 0.06, "quad 5 bidders, 1 item, lr = 0.06")
    plot_results(data7, theo_lr, "quad 10 bidders, 1 item, lr = " + str(theo_lr))
    plot_results(data8, theo_lr, "quad 10 bidders, 4 items, lr = " + str(theo_lr))


def part2():

    # uniform [0,1]
    data2_1 = gen_data2(n1, k1, 2, [np.random.random_sample] * 2)
    plot_results(data2_1, theo_lr, "uni 2 participants, lr = " + str(theo_lr), h=2)
    plot_results(data2_1, 0.15, "uni 2 participants, lr = 0.15", h=2)

    data2_2 = gen_data2(n1, k1 * 3, 3, [np.random.random_sample] * 3)
    plot_results(data2_2, theo_lr, "uni 3 participants, lr = " + str(theo_lr), h=3)

    data2_3 = gen_data2(n1, k1 * 4, 4, [np.random.random_sample] * 4)
    plot_results(data2_3, theo_lr, "uni 4 participants, lr = " + str(theo_lr), h=4)

    # quadratic [0,1]
    data2_4 = gen_data2(n1, k1, 2, [quadratic_dist] * 2)
    plot_results(data2_4, theo_lr, "quad 2 participants, lr = " + str(theo_lr), h=2)
    plot_results(data2_4, 0.15, "quad 2 participants, lr = 0.15", h=2)

    # one uniform, one quadratic [0,1]
    data2_5 = gen_data2(n1, k1, 2, [np.random.random_sample, quadratic_dist])
    plot_results(data2_5, theo_lr, "uni/quad 2 participants, lr = " + str(theo_lr), h=2)
    plot_results(data2_5, 0.15, "uni/quad 2 participants, lr = 0.15", h=2)


if __name__ == '__main__':
    n1 = 10000
    k1 = 100
    theo_lr = math.sqrt(math.log(k1) / n1)

    print("\nPart 1\n")
    part1()
    print("\nPart 2\n")
    part2()

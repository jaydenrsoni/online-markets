import numpy as np
import math


def online_competition(mat, n, lr, alg1, alg2, print_out=True, verbose=False):

    k = len(mat)                # assumes symmetric number of actions for both players

    action_payoffs1 = [0] * k   # V vector holding cumulative payoffs of each action
    action_payoffs2 = [0] * k
    actions1 = []               # list of actions algorithm takes
    actions2 = []
    total_payoff1 = 0           # payoff of actions algorithm takes
    total_payoff2 = 0

    max_payoffs1 = [max([mat[i][j][0] for j in range(k)]) for i in range(k)]    # max payoffs (h) for each action
    max_payoffs2 = [max([mat[j][i][1] for j in range(k)]) for i in range(k)]

    if alg1 == "ftpl":
        hallucinations1 = np.random.geometric(p=lr, size=k) * max(max_payoffs1)
    elif alg1 == "ftpl_nonuniform":
        hallucinations1 = np.multiply(np.random.geometric(p=lr, size=k), max_payoffs1)

    if alg2 == "ftpl":
        hallucinations2 = np.random.geometric(p=lr, size=k) * max(max_payoffs2)
    elif alg2 == "ftpl_nonuniform":
        hallucinations2 = np.multiply(np.random.geometric(p=lr, size=k), max_payoffs2)

    # simulate n rounds
    for _ in range(n):

        # choose player 1's action action using given algorithm
        if alg1 == "ew":
            raw_probabilities = [math.pow((1 + lr), action_payoffs1[i] / max(max_payoffs1)) for i in range(k)]
            action1 = ew_action(raw_probabilities, verbose)

        elif alg1 == "ew_nonuniform":
            raw_probabilities = [math.pow((1 + lr), action_payoffs1[i] / max_payoffs1[i]) for i in range(k)]
            action1 = ew_action(raw_probabilities, verbose)

        elif alg1 == "ftpl" or alg1 == "ftpl_nonuniform":
            action1 = np.argmax(np.add(hallucinations1, action_payoffs1))

        elif alg1 == "hold_right":
            action1 = np.argmax(max_payoffs1)

        elif alg1 == "random":
            action1 = np.random.choice(range(k))

        elif alg1 == "reactive":
            mapping = {0: 2, 1: 0, 2: 2}
            if not actions2:
                action1 = 2
            else:
                action1 = mapping[action2]

        else:
            raise Exception("not a valid algorithm")

        actions1.append(action1)

        # choose player 2's action action using given algorithm
        if alg2 == "ew":
            raw_probabilities = [math.pow((1 + lr), action_payoffs2[i] / max(max_payoffs2)) for i in range(k)]
            action2 = ew_action(raw_probabilities, verbose)

        elif alg2 == "ew_nonuniform":
            raw_probabilities = [math.pow((1 + lr), action_payoffs2[i] / max_payoffs2[i]) for i in range(k)]
            action2 = ew_action(raw_probabilities, verbose)

        elif alg2 == "ftpl" or alg2 == "ftpl_nonuniform":
            action2 = np.argmax(np.add(hallucinations2, action_payoffs2))

        elif alg2 == "hold_right":
            action2 = np.argmax(max_payoffs2)

        elif alg2 == "random":
            action2 = np.random.choice(range(k))

        else:
            raise Exception("not a valid algorithm")

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

    convergence1 = sum(actions1[n - int(n/20):])/(n/20)
    convergence2 = sum(actions2[n - int(n/20):])/(n/20)

    if print_out:
        print("Player1: ")
        print("total payoff: ", total_payoff1)
        print('convergence: ', convergence1)

        print("Player2: ")
        print("total payoff: ", total_payoff2)
        print('convergence: ', convergence2)

    return total_payoff1, total_payoff2, convergence1, convergence2


def run_simulations(num_trials, num_rounds, pay_mat, alg1, alg2, print_option=False):

    wins = [0, 0]
    action_convergence = {0: [0, 0], 1: [0, 0]}
    overall_payoff1 = 0
    overall_payoff2 = 0

    for _ in range(num_trials):
        payoff1, payoff2, action1, action2 = online_competition(pay_mat, num_rounds,
                                                                theoretical_lr(num_rounds, len(pay_mat)),
                                                                alg1, alg2, print_out=print_option)
        
        winner = int(payoff2 > payoff1)   # 0 if player 1 wins, 1 if player 2 wins

        wins[winner] += 1
        action_convergence[winner][0] += action1
        action_convergence[winner][1] += action2
        overall_payoff1 += payoff1/num_rounds
        overall_payoff2 += payoff2/num_rounds

    if wins[0] > 0:
        action_convergence[0] = tuple([round(x, 4) for x in np.divide(action_convergence[0], wins[0])])
    else:
        action_convergence[0] = None

    if wins[1] > 0:
        action_convergence[1] = tuple([round(x, 4) for x in np.divide(action_convergence[1], wins[1])])
    else:
        action_convergence[1] = None

    print("\nPayoff Matrix: ")
    for row in pay_mat:
        print(row)
    print("Algorithm 1: ", alg1)
    print("Algorithm 2: ", alg2)
    print("Equilibrium: ", action_convergence)
    print("Wins: ", wins)
    print("Average Payoffs: ", (round(overall_payoff1/num_trials, 4), round(overall_payoff2/num_trials, 4)))


def ew_action(rp, verb):
    norm_probabilities = np.divide(rp, math.fsum(rp))
    action = np.random.choice(len(norm_probabilities), 1, p=norm_probabilities)[0]
    if verb:
        print("player probabilities: ", norm_probabilities)
    return action


def theoretical_lr(n, k):
    return math.sqrt(math.log(k) / n)


def part1():
    print("\nPart 1...\n\n")
    print("\nSame algorithm for both players...\n")
    run_simulations(number_of_trials, nrounds, bots1, "ew", "ew")
    run_simulations(number_of_trials, nrounds, bots2, "ew", "ew")
    run_simulations(number_of_trials, nrounds, bots3, "ew", "ew")
    run_simulations(number_of_trials, nrounds, bots4, "ew", "ew")

    run_simulations(number_of_trials, nrounds, bots1, "ftpl", "ftpl")
    run_simulations(number_of_trials, nrounds, bots2, "ftpl", "ftpl")
    run_simulations(number_of_trials, nrounds, bots3, "ftpl", "ftpl")
    run_simulations(number_of_trials, nrounds, bots4, "ftpl", "ftpl")

    run_simulations(number_of_trials, nrounds, bots1, "ew_nonuniform", "ew_nonuniform")
    run_simulations(number_of_trials, nrounds, bots2, "ew_nonuniform", "ew_nonuniform")
    run_simulations(number_of_trials, nrounds, bots3, "ew_nonuniform", "ew_nonuniform")
    run_simulations(number_of_trials, nrounds, bots4, "ew_nonuniform", "ew_nonuniform")

    run_simulations(number_of_trials, nrounds, bots1, "ftpl_nonuniform", "ftpl_nonuniform")
    run_simulations(number_of_trials, nrounds, bots2, "ftpl_nonuniform", "ftpl_nonuniform")
    run_simulations(number_of_trials, nrounds, bots3, "ftpl_nonuniform", "ftpl_nonuniform")
    run_simulations(number_of_trials, nrounds, bots4, "ftpl_nonuniform", "ftpl_nonuniform")

    print("\nUniform vs. nonuniform bounds on symmetric games...\n")
    run_simulations(number_of_trials, nrounds, bots1, "ew", "ew_nonuniform")
    run_simulations(number_of_trials, nrounds, bots2, "ew", "ew_nonuniform")
    run_simulations(number_of_trials, nrounds, bots1, "ftpl", "ftpl_nonuniform")
    run_simulations(number_of_trials, nrounds, bots2, "ftpl", "ftpl_nonuniform")

    print("\nExponential weights vs. follow the perturbed leader on symmetric games...\n")
    run_simulations(number_of_trials, nrounds, bots1, "ew", "ftpl")
    run_simulations(number_of_trials, nrounds, bots2, "ew", "ftpl")

    print("\nExponential weights vs. follow the perturbed leader (nonuniform bounds) on symmetric games...\n")
    run_simulations(number_of_trials, nrounds, bots1, "ew", "ftpl_nonuniform")
    run_simulations(number_of_trials, nrounds, bots2, "ew", "ftpl_nonuniform")

    print("\nExponential weights vs. follow the perturbed leader on asymmetric games...\n")
    run_simulations(number_of_trials, nrounds, bots3, "ew", "ftpl")
    run_simulations(number_of_trials, nrounds, bots3, "ftpl", "ew")
    run_simulations(number_of_trials, nrounds, bots4, "ew", "ftpl")
    run_simulations(number_of_trials, nrounds, bots4, "ftpl", "ew")

    print("\nExponential weights vs. follow the perturbed leader (nonuniform bounds) on asymmetric games...\n")
    run_simulations(number_of_trials, nrounds, bots3, "ew", "ftpl_nonuniform")
    run_simulations(number_of_trials, nrounds, bots3, "ftpl_nonuniform", "ew")
    run_simulations(number_of_trials, nrounds, bots4, "ew", "ftpl_nonuniform")
    run_simulations(number_of_trials, nrounds, bots4, "ftpl_nonuniform", "ew")

    print("\nAssurance games...\n")
    run_simulations(number_of_trials, nrounds, assurance1, "ftpl_nonuniform", "ew")
    run_simulations(number_of_trials, nrounds, assurance1, "ew", "ftpl_nonuniform")
    run_simulations(number_of_trials, nrounds, assurance2, "ftpl_nonuniform", "ew")
    run_simulations(number_of_trials, nrounds, assurance2, "ew", "ftpl_nonuniform")
    run_simulations(number_of_trials, nrounds, assurance3, "ftpl_nonuniform", "ew")
    run_simulations(number_of_trials, nrounds, assurance3, "ew", "ftpl_nonuniform")

    print("\nCompetitive game...\n")
    run_simulations(number_of_trials, nrounds, comp1, "ew", "ew")
    run_simulations(number_of_trials, nrounds, comp1, "ftpl_nonuniform", "ftpl_nonuniform")
    run_simulations(number_of_trials, nrounds, comp1, "ftpl_nonuniform", "ew")


def part2():
    print("\nPart 2...\n\n")
    print("\nBattle of the sexes...\n")
    run_simulations(number_of_trials, nrounds, bots1, "hold_right", "ew")
    run_simulations(number_of_trials, nrounds, bots2, "hold_right", "ew")
    run_simulations(number_of_trials, nrounds, bots3, "hold_right", "ew")
    run_simulations(number_of_trials, nrounds, bots3, "ew", "hold_right")
    run_simulations(number_of_trials, nrounds, bots4, "hold_right", "ew")
    run_simulations(number_of_trials, nrounds, bots4, "ew", "hold_right")

    run_simulations(number_of_trials, nrounds, bots1, "hold_right", "ftpl_nonuniform")
    run_simulations(number_of_trials, nrounds, bots2, "hold_right", "ftpl_nonuniform")
    run_simulations(number_of_trials, nrounds, bots3, "hold_right", "ftpl_nonuniform")
    run_simulations(number_of_trials, nrounds, bots3, "ftpl_nonuniform", "hold_right")
    run_simulations(number_of_trials, nrounds, bots4, "hold_right", "ftpl_nonuniform")
    run_simulations(number_of_trials, nrounds, bots4, "ftpl_nonuniform", "hold_right")

    print("\nAssurance games...\n")
    run_simulations(number_of_trials, nrounds, assurance1, "hold_right", "ew")
    run_simulations(number_of_trials, nrounds, assurance1, "ew", "hold_right")
    run_simulations(number_of_trials, nrounds, assurance2, "hold_right", "ew")
    run_simulations(number_of_trials, nrounds, assurance2, "ew", "hold_right")
    run_simulations(number_of_trials, nrounds, assurance3, "hold_right", "ew")
    run_simulations(number_of_trials, nrounds, assurance3, "ew", "hold_right")

    run_simulations(number_of_trials, nrounds, assurance1, "hold_right", "ftpl_nonuniform")
    run_simulations(number_of_trials, nrounds, assurance1, "ftpl_nonuniform", "hold_right")
    run_simulations(number_of_trials, nrounds, assurance2, "hold_right", "ftpl_nonuniform")
    run_simulations(number_of_trials, nrounds, assurance2, "ftpl_nonuniform", "hold_right")
    run_simulations(number_of_trials, nrounds, assurance3, "hold_right", "ftpl_nonuniform")
    run_simulations(number_of_trials, nrounds, assurance3, "ftpl_nonuniform", "hold_right")

    print("\nCompetitive game...\n")
    run_simulations(number_of_trials, nrounds, comp1, "hold_right", "ew")
    run_simulations(number_of_trials, nrounds, comp1, "random", "ew")
    run_simulations(number_of_trials, nrounds, comp1, "reactive", "ew")

    run_simulations(number_of_trials, nrounds, comp1, "hold_right", "ftpl_nonuniform")
    run_simulations(number_of_trials, nrounds, comp1, "random", "ftpl_nonuniform")
    run_simulations(number_of_trials, nrounds, comp1, "reactive", "ftpl_nonuniform")


def supplement():
    # Part 1
    run_simulations(number_of_trials, nrounds, bots3, "ew", "ew_nonuniform")
    run_simulations(number_of_trials, nrounds, bots3, "ew_nonuniform", "ew")
    run_simulations(number_of_trials, nrounds, bots4, "ew", "ew_nonuniform")
    run_simulations(number_of_trials, nrounds, bots4, "ew_nonuniform", "ew")

    run_simulations(number_of_trials, nrounds, bots3, "ftpl", "ftpl_nonuniform")
    run_simulations(number_of_trials, nrounds, bots3, "ftpl_nonuniform", "ftpl")
    run_simulations(number_of_trials, nrounds, bots4, "ftpl", "ftpl_nonuniform")
    run_simulations(number_of_trials, nrounds, bots4, "ftpl_nonuniform", "ftpl")


if __name__ == "__main__":

    # payoff matrices indexed as follows: bots[row][col][player]

    # (10, 5)   (0, 0)
    # (0, 0)    (5, 10)
    bots1 = (((10, 5), (0, 0)), ((0, 0), (5, 10)))

    # (5, 1)   (0, 0)
    # (0, 0)   (1, 5)
    bots2 = (((5, 1), (0, 0)), ((0, 0), (1, 5)))

    # (5, 2)   (0, 0)
    # (0, 0)   (2, 4)
    bots3 = (((5, 2), (0, 0)), ((0, 0), (2, 4)))

    # (8, 2)   (0, 0)
    # (0, 0)   (2, 4)
    bots4 = (((8, 2), (0, 0)), ((0, 0), (2, 4)))

    assurance1 = (((10, 10), (0, 0)), ((0, 0), (5, 5)))
    assurance2 = (((100, 100), (0, 0)), ((0, 0), (5, 5)))
    assurance3 = (((100, 100), (0, 0)), ((0, 0), (99, 99)))

    # (2, 2)   (4, 0)    (4, 0)  
    # (0, 4)   (3, 3)    (5, 1)
    # (0, 4)   (1, 5)    (4, 4)
    comp1 = (((2,2), (4,0), (4,0)), ((0, 4), (3,3), (5,1)), ((0, 4), (1, 5), (4, 4)))

    number_of_trials = 100
    nrounds = 1000

    # part1()
    # part2()
    supplement()


# the below functions are precursors to the online_competition function illustrating each of the 4 algorithms separately

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
        raw_probabilities = [math.pow((1 + lr), action_payoffs1[j] / h) for j in range(k)]
        norm_probabilities = np.divide(raw_probabilities, math.fsum(raw_probabilities))
        action1 = np.random.choice(k, 1, p=norm_probabilities)[0]
        if verbose:
            print("player 1 probabilities: ", norm_probabilities)

        actions1.append(action1)

        # player 2
        raw_probabilities = [math.pow((1 + lr), action_payoffs2[j] / h) for j in range(k)]
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


def double_exponential_weights_nonuniform_bounds(n, k, lr, mat, verbose=True):
    action_payoffs1 = [0] * k  # V vector holding cumulative payoffs of each action
    action_payoffs2 = [0] * k  # V vector holding cumulative payoffs of each action
    actions1 = []  # list of actions algorithm takes
    actions2 = []  # list of actions algorithm takes
    total_payoff1 = 0  # payoff of actions algorithm takes
    total_payoff2 = 0  # payoff of actions algorithm takes

    # simulate n rounds
    for _ in range(n):

        # player 1
        raw_probabilities = [math.pow((1 + lr), action_payoffs1[i] /
                                      max([mat[i][j][0] for j in range(k)])) for i in range(k)]
        norm_probabilities = np.divide(raw_probabilities, math.fsum(raw_probabilities))
        action1 = np.random.choice(k, 1, p=norm_probabilities)[0]
        if verbose:
            print("player 1 probabilities: ", norm_probabilities)

        actions1.append(action1)

        # player 2
        raw_probabilities = [math.pow((1 + lr), action_payoffs2[i] /
                                      max([mat[j][i][1] for j in range(k)])) for i in range(k)]
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
                                  [max([mat[i][j][0] for j in range(k)]) for i in range(k)])
    hallucinations2 = np.multiply(np.random.geometric(p=lr, size=k),
                                  [max([mat[j][i][1] for j in range(k)]) for i in range(k)])

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

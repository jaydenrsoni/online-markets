def dataset1():
    return 0


def dataset2():
    return 0


def exponential_weights(data, lr):
    regret = 0
    print(regret)


def follow_the_perturbed_leader(data, lr):
    regret = 0
    print(regret)


if __name__ == '__main__':
    # theoretically optimal learning rate
    exponential_weights(dataset1())
    follow_the_perturbed_leader(dataset1())

    # empirically optimal learning rate
    exponential_weights(dataset1())
    follow_the_perturbed_leader(dataset1())

    # theoretically optimal learning rate
    exponential_weights(dataset2())
    follow_the_perturbed_leader(dataset2())

    # empirically optimal learning rate
    exponential_weights(dataset2())
    follow_the_perturbed_leader(dataset2())

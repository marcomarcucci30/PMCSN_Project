import statistics

from matplotlib import pyplot as plt

from utils.rngs import random
from utils.rvgs import TruncatedNormal
from utils.rvms import idfNormal


def online_variance(n, mean, variance, x):
    delta = x - mean
    variance = variance + delta * delta * (n - 1) / n
    mean = mean + delta / n
    return mean, variance


if __name__ == '__main__':
    # list = [1, 6.2, 3, None, 9, 2, 6.1]
    # list = sorted(list, key=lambda x: (x is None, x))
    ## list.sort()
    # print(list)
    # a = 17.7
    # print(a // 1)
    # print(1 + 5.0)

    # nodes = 5
    # for i in range(0, 10):
    #    for j in range(0, nodes):
    #        print("ciao")
    #    print("\n")
    #    nodes = 2 + i
    # rep = 5
    # l = rep / (rep + 1.0)
    # print(l)
    # avg = 5.0
    # nodes = 11
    # print(avg / (nodes - 1.0))

    # ciao = []
    # count = 0
    # ble = 0
    # for i in range(0, 10):
    #    conf_dict = {
    #        "delay_arcades": 0.0,
    #        "delay_system": 0.0,
    #        "conf": (0, 0, 0, 0)
    #    }
    #    ciao.append(conf_dict)
    #    ciao[i]["delay_arcades"] = i + 0.0

    # print(ciao)

    ## x = [1, 2, 3]
    # y = [4, 8, 6]
    # list1 = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    # x = [str(i) for i in list1]  # in 0 global stats

    # plt.plot(x, y)

    # plt.legend(["Gain"])
    # plt.title("Gain")
    # plt.xlabel("Configuration")
    # plt.ylabel("Gain function")
    # plt.show()

    list_gauss = []
    n = 0
    mean = 0.0
    var = 0.0

    for i in range(0, 400):
        n += 1
        r = random()
        list_gauss.append(TruncatedNormal(15.0, 3.0, 3.0, 25.0))
        mean, var = online_variance(n, mean, var, list_gauss[i])
    print(mean, var)
    print(statistics.variance(list_gauss))
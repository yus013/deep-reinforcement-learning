import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    logfile = ''
    for root, dirs, files in os.walk('log'):
        logfile = os.path.join(root, files[0])

    t = list()
    eps = list()
    score = list()
    loss1 = list()
    loss2 = list()
    for line in open(logfile, 'r'):
        info = line.strip().split()
        if info[2] == 'nan':
            continue

        t.append(float(info[0]) / 1e6)
        eps.append(float(info[1]))
        score.append(float(info[2]))
        loss1.append(float(info[3]))
        loss2.append(float(info[4]))


    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.title("$\epsilon$-greedy")
    plt.grid()
    plt.plot(t, eps)

    plt.subplot(2, 2, 2)
    plt.title("Average Score")
    plt.grid()
    plt.plot(t, score)

    plt.subplot(2, 2, 3)
    plt.title("Loss1")
    plt.grid()
    plt.plot(t, loss1)

    plt.subplot(2, 2, 4)
    plt.title("Loss2")
    plt.grid()
    plt.plot(t, loss2)

    plt.show()

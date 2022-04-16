import datafunctions
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np



if __name__ == '__main__':





    plt.plot(x,y, lw=1.5)

    plt.xlabel("FPR", fontsize=15)
    plt.ylabel("TPR", fontsize=15)
    plt.title("ROC")
    plt.legend(loc="lower right")
    plt.show()
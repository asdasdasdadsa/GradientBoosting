import numpy as np
import dt


class AD():

    def __init__(self, M, alfa):
        self.M = M
        self.y = []
        self.node = []
        self.al = []
        self.alfa = alfa

    def decision_stump(self, dn, cl):
        d = dt.DT(1, 0.05, 5, 2)
        root = dt.Node()
        d.build_tree(dn, cl, root, 0)
        return d, root

    def gradBoost(self, dn, cl):
        self.Y = [sum(cl) / len(cl)]
        for i in range(self.M):
            r = cl - self.alfa * np.sum(
                np.array([self.Y[i].pass_tree_all(self.node[i - 1], dn) for i in range(1, len(self.Y))])) - self.Y[0]
            d, root = self.decision_stump(dn, r)
            self.Y.append(d)
            self.node.append(root)

    def pass_ad(self, dn):
        s = self.Y[0] + self.alfa * np.sum(
            np.array([self.Y[i].pass_tree(self.node[i - 1], dn) for i in range(1, len(self.Y))]))
        return s
        # if np.sum(np.array(self.al) * np.array([self.y[j].pass_tree(self.node[j], dn) for j in range(len(self.y))])) >= 0:
        #     return 1
        # else:
        #     return -1

    def pass_ad_all(self, dn):
        s = []
        for i in range(len(dn)):
            s.append(self.pass_ad(dn[i]))
        return np.array(s)

    def MSE(self, dn, cl):
        s = np.sum((cl - self.pass_ad_all(dn))**2) / len(dn)
        return s

import cv2
import numpy as np
import math
import queue
# import matplotlib.pyplot as plt
import random
import time


class GCEngine:
    def __init__(self, img):
        self.img = img
        self.img_shape = img.shape

    def OriginalIterate(self, p1, p2):
        self.x1, self.y1, self.x2, self.y2 = p1[0], p1[1], p2[0], p2[1]
        self.alpha = np.zeros((self.img_shape[0], self.img_shape[1]))
        self.alpha[self.x1:self.x2 + 1, self.y1:self.y2 + 1] = 1
        self.init_V()
        self.init_GMM()
        max_iteration = 10
        for i in range(max_iteration):
            self.assign_k()
            self.update_GMM()
            self.graphcut()  # to be implemented

        return self.alpha

    def add_foreground(self, pixels):
        # to be implemented
        return

    def add_background(self, pixels):
        # to be implemented
        return

    def rerun(self):
        # to be implemented
        return np.zeros(self.img_size, np.uint8)

    def init_GMM(self):
        # GMM parameters
        self.k = np.zeros_like(self.alpha)
        self.K = 5
        self.co = np.zeros((2, self.K))
        self.mu = np.zeros((2, self.K, 3))
        self.Siginv = np.zeros((2, self.K, 3, 3))
        self.sig = np.ones((2, self.K))
        self.co[:, :] = 1 / (self.K)
        self.Siginv[:, :] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # divide object and background pixels
        mask = np.array(self.alpha == 0)
        A = self.img[mask]  # background pixels
        B = self.img[~mask]  # object pixels

        # initialize with k-means
        index1 = np.random.choice(range(A.shape[0]), self.K, replace=False)
        index2 = np.random.choice(range(B.shape[0]), self.K, replace=False)
        self.mu[0] = A[index1]
        self.mu[1] = B[index2]
        # A_uniq=np.unique(A,axis=0)
        # B_uniq=np.unique(B,axis=0)
        # if A_uniq.shape[0]>self.K:
        # index1 = np.random.choice(range(A_uniq.shape[0]), self.K, replace=False)
        # self.mu[0] = A_uniq[index1]
        # if B_uniq.shape[0]>self.K:
        # index2 = np.random.choice(range(B_uniq.shape[0]), self.K, replace=False)
        # self.mu[1] = B_uniq[index2]

        # begin iteration
        max_iteration = 10
        for i in range(max_iteration):
            A_group = [[] for j in range(self.K)]
            B_group = [[] for j in range(self.K)]

            # group
            for j in range(A.shape[0]):
                distance = np.sum((A[j] - self.mu[0]) ** 2, axis=1)
                type = np.argmin(distance)
                A_group[type].append(A[j])
            for j in range(B.shape[0]):
                distance = np.sum((B[j] - self.mu[1]) ** 2, axis=1)
                type = np.argmin(distance)
                B_group[type].append(B[j])

            self.update(A_group, B_group)

    # A_group is a 2-dimensional list of the pixels in the background,B_group is a 2-dimensional list of the grouped pixels in the object
    def update(self, A_group, B_group):
        sum1 = sum([len(A_group[i]) for i in range(self.K)])
        sum2 = sum([len(B_group[i]) for i in range(self.K)])
        for j in range(self.K):
            n1 = np.array(A_group[j])
            n2 = np.array(B_group[j])
            self.co[0][j] = len(A_group[j]) / sum1 + 1e-10
            if n1.shape[0] != 0:
                d1 = math.fabs(np.linalg.det(n1.T @ n1 / n1.shape[0]))
                if d1 > 1e-4:
                    self.mu[0][j] = np.mean(n1, axis=0)
                    self.Siginv[0][j] = np.linalg.inv(n1.T @ n1 / n1.shape[0])
                    self.sig[0][j] = math.sqrt(d1)
            self.co[1][j] = len(B_group[j]) / sum2 + 1e-10
            if n2.shape[0] != 0:
                d2 = math.fabs(np.linalg.det(n2.T @ n2 / n2.shape[0]))
                if d2 > 1e-4:
                    self.mu[1][j] = np.mean(n2, axis=0)
                    self.Siginv[1][j] = np.linalg.inv(n2.T @ n2 / n2.shape[0])
                    self.sig[1][j] = math.sqrt(d2)

    def assign_k(self):
        for i in range(self.img_shape[0]):
            for j in range(self.img_shape[1]):
                if self.alpha[i, j] == 0:
                    A = [-math.log(self.co[0, z]) + math.log(self.sig[0, z]) + (
                                0.5 * (self.img[i, j] - self.mu[0, z]) @ self.Siginv[0, z] @ (
                            (self.img[i, j] - self.mu[0, z]).reshape(-1, 1)))[0] for z in range(self.K)]
                    P = np.array(A)
                    self.k[i, j] = np.argmin(P)
                else:
                    B = [-math.log(self.co[1, z]) + math.log(self.sig[1, z]) + (
                                0.5 * (self.img[i, j] - self.mu[1, z]) @ self.Siginv[1, z] @ (
                            (self.img[i, j] - self.mu[1, z]).reshape(-1, 1)))[0] for z in range(self.K)]
                    P = np.array(B)
                    self.k[i, j] = np.argmin(P)

    def update_GMM(self):
        A_group = [[] for j in range(self.K)]
        B_group = [[] for j in range(self.K)]
        for i in range(self.img_shape[0]):
            for j in range(self.img_shape[1]):
                if self.alpha[i, j] == 0:
                    A_group[self.k[i, j].astype('int')].append(self.img[i, j])
                else:
                    B_group[self.k[i, j].astype('int')].append(self.img[i, j])
        self.update(A_group, B_group)

    def init_V(self):
        self.gama = 50
        a = 0
        dx = [-1, 0, 0, 1]
        dy = [0, 1, -1, 0]
        for i in range(self.img_shape[0]):
            for j in range(self.img_shape[1]):
                for z in range(4):
                    nx = i + dx[z]
                    ny = j + dy[z]
                    if 0 <= nx < self.img_shape[0] and 0 <= ny < self.img_shape[1]:
                        a += np.sum((self.img[i, j] - self.img[nx, ny]) ** 2)
        a /= self.img_shape[0] * self.img_shape[1] * 4
        if a != 0:
            self.beta = 1 / a
        else:
            self.beta = 0

    def graphcut(self):
        vertex_num = self.img_shape[0] * self.img_shape[1] + 2
        g = graph(vertex_num)

        dx = [-1, 0, 0, 1]
        dy = [0, 1, -1, 0]
        o = 0
        for i in range(self.img_shape[0]):
            for j in range(self.img_shape[1]):
                A = [-math.log(self.co[0, z]) + math.log(self.sig[0, z]) + (
                        0.5 * (self.img[i, j] - self.mu[0, z]) @ self.Siginv[0, z] @ (
                    (self.img[i, j] - self.mu[0, z]).reshape(-1, 1)))[0] for z in range(self.K)]
                U0 = np.array(A)
                u0 = np.min(U0).astype('int')
                if (u0 < o):
                    o = u0
                B = [-math.log(self.co[1, z]) + math.log(self.sig[1, z]) + (
                        0.5 * (self.img[i, j] - self.mu[1, z]) @ self.Siginv[1, z] @ (
                    (self.img[i, j] - self.mu[1, z]).reshape(-1, 1)))[0] for z in range(self.K)]
                U1 = np.array(B)
                u1 = np.min(U1).astype('int')
                if (u1 < o):
                    o = u1
        for i in range(self.img_shape[0]):
            for j in range(self.img_shape[1]):
                A = [-math.log(self.co[0, z]) + math.log(self.sig[0, z]) + (
                        0.5 * (self.img[i, j] - self.mu[0, z]) @ self.Siginv[0, z] @ (
                    (self.img[i, j] - self.mu[0, z]).reshape(-1, 1)))[0] for z in range(self.K)]
                U0 = np.array(A)
                u0 = np.min(U0).astype('int')
                g.addedge(0, i * self.img_shape[1] + j + 1, u0 - o)
                g.addedge(i * self.img_shape[1] + j + 1, 0, 0)
                B = [-math.log(self.co[1, z]) + math.log(self.sig[1, z]) + (
                        0.5 * (self.img[i, j] - self.mu[1, z]) @ self.Siginv[1, z] @ (
                    (self.img[i, j] - self.mu[1, z]).reshape(-1, 1)))[0] for z in range(self.K)]
                U1 = np.array(B)
                u1 = np.min(U1).astype('int')
                g.addedge(i * self.img_shape[1] + j + 1, vertex_num - 1, u1 - o)
                g.addedge(vertex_num - 1, i * self.img_shape[1] + j + 1, 0)
                for z in range(4):
                    nx = i + dx[z]
                    ny = j + dy[z]
                    if 0 <= nx < self.img_shape[0] and 0 <= ny < self.img_shape[1]:
                        if self.alpha[i, j] != self.alpha[nx, ny]:
                            v = self.gama * np.exp(-self.beta * np.sum((self.img[i, j] - self.img[nx, ny]) ** 2))
                            g.addedge(i * self.img_shape[1] + j + 1, nx * self.img_shape[1] + ny + 1, v)
        a = g.dinic()
        i = g.head[0]
        while i != -1:
            x = (g.edges[i].to - 1) // self.img_shape[1]
            y = (g.edges[i].to - 1) % self.img_shape[1]
            if g.edges[i].w == 0:
                self.alpha[x, y] = 0
            else:
                self.alpha[x, y] = 1
            i = g.edges[i].nex
        c = self.alpha
        print(c)


class edge:
    def __init__(self, a, b, c):
        self.to = a
        self.w = b
        self.nex = c


class graph:
    def __init__(self, vertex_num):
        self.head = [-1 for i in range(vertex_num)]
        self.dis = [-1 for i in range(vertex_num)]
        self.edges = []
        self.n = vertex_num
        self.flow = 0

    def addedge(self, source, destination, weight):
        e = edge(destination, weight, self.head[source])
        self.head[source] = len(self.edges)
        self.edges.append(e)

    def bfs(self):
        self.dis = [-1 for i in range(self.n)]
        self.dis[0] = 0
        q = queue.Queue()
        q.put(0)
        while (not q.empty()) and self.dis[self.n - 1] == -1:
            u = q.get()
            c_e = self.head[u]
            while c_e != -1:
                _to = self.edges[c_e].to
                if self.dis[_to] == -1 and self.edges[c_e].w > 0:
                    self.dis[_to] = self.dis[u] + 1
                    q.put(_to)
                c_e = self.edges[c_e].nex
        return self.dis[self.n - 1] != -1

    def dfs(self, u, limit):
        if u == self.n - 1:
            return limit
        flow1 = 0
        c_e = self.head[u]
        while c_e != -1 and limit > 0:
            _to = self.edges[c_e].to
            if self.dis[_to] == self.dis[u] + 1 and self.edges[c_e].w > 0:
                flow2 = self.dfs(_to, min(limit, self.edges[c_e].w))
                flow1 += flow2
                self.edges[c_e].w -= flow2
                self.edges[c_e ^ 1].w += flow2
                limit -= flow2
            c_e = self.edges[c_e].nex
        return flow1

    # 默认第一个点是源点，最后一个点是汇点
    def dinic(self):
        self.flow = 0
        while self.bfs():
            self.flow += self.dfs(0, 100000000000000)
        return self.flow



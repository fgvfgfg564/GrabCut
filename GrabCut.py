import cv2
import numpy as np
import math
import queue
# import matplotlib.pyplot as plt
import random
import time


class kmeans:
    def __init__(self, data, dim, n, max_iter):
        self.row = int(data.size / dim)
        self.dim = dim
        self.data = data.reshape(self.row, dim).copy().astype(np.float)
        self.n = n
        self._init_center()
        self.type = np.zeros(self.row, dtype=np.uint)
        self.max_iter = max_iter

    def _init_center(self):
        index1 = np.random.choice(range(self.row), self.n, replace=False)
        self.center = self.data[index1]

    def assign_type(self):
        for i in range(self.row):
            self.type[i] = np.argmin(np.sum((self.data[i] - self.center) ** 2, axis=1))

    def update_center(self):
        cluster_length = np.array([np.where(self.type == i)[0].size for i in range(self.n)])
        for i in range(self.n):
            if cluster_length[i] > 0:
                self.center[i] = np.sum(self.data[np.where(self.type == i)], axis=0) / cluster_length[i]
            else:
                t = np.argmax(cluster_length)
                p = np.where(self.type == t)
                pixels = self.data[p]
                distance_center = np.sum((pixels - self.center[t]) ** 2, axis=1)
                u = np.argmax(distance_center)
                u = p[0][u]
                self.type[u] = i
                self.center[i] = self.data[u]

    def run(self):
        for i in range(self.max_iter):
            self.assign_type()
            self.update_center()

    def output(self):
        self.data_by_comp = np.array([self.data[np.where(self.type == i)] for i in range(self.n)])
        return self.data_by_comp


class GMM:
    def __init__(self, k=5):
        self.k = k
        self.pi = np.zeros(k)
        self.mu = np.zeros((k, 3))
        self.cov = np.zeros((k, 3, 3))
        self.cov_inv = np.zeros((k, 3, 3))
        self.cov_det = np.zeros(k)
        self.pixel_count = np.zeros(k)
        self.pixel_total_count = 0
        self._sum = np.zeros((k, 3))
        self._prob = np.zeros((k, 3, 3))

    def prob_pixel_component(self, pixel, i):
        p = pixel - self.mu[i]
        x = (p @ self.cov_inv[i] @ (p.reshape(-1, 1)))[0]
        return 1 / np.sqrt(self.cov_det[i]) * np.exp(-0.5 * x)

    def prob_pixel_GMM(self, pixel):
        return sum([self.pi[i] * self.prob_pixel_component(pixel, i) for i in range(self.k)])

    def most_likely_pixel_component(self, pixel):
        a = np.array([self.prob_pixel_component(pixel, i) for i in range(self.k)])
        return np.argmax(a)

    def max_D_pixel_component(self, pixel):
        a = np.array([self.pi[i] * self.prob_pixel_component(pixel, i) for i in range(self.k)])
        return np.argmax(a)

    def add_pixel(self, pixel, i):
        self._sum[i] += pixel
        self._prob[i] += pixel.reshape(-1, 1) @ pixel.reshape(1, -1)
        self.pixel_count[i] += 1
        self.pixel_total_count += 1

    def update(self):
        variance = 0.01
        for i in range(self.k):
            n = self.pixel_count[i]
            if n == 0:
                self.pi[i] = 0
            else:
                self.pi[i] = n / self.pixel_total_count
                self.mu[i] = self._sum[i] / n
                self.cov[i] = self._prob[i] / n - self.mu[i].reshape(-1, 1) @ self.mu[i].reshape(1, -1)
                self.cov_det[i] = np.linalg.det(self.cov[i])
                while self.cov_det[i] <= 0:
                    self.cov[i] += np.diag([variance for j in range(3)])
                    self.cov_det[i] = np.linalg.det(self.cov[i])
                self.cov_inv[i] = np.linalg.inv(self.cov[i])

    def clear(self):
        self._sum[:, :] = 0
        self._prob[:, :, :] = 0
        self.pixel_count[:] = 0
        self.pixel_total_count = 0


class GCEngine:
    def __init__(self, img):
        self.img = img.astype(np.float)
        self.img_shape = img.shape
        self.row = self.img_shape[0]
        self.col = self.img_shape[1]

        self.k = 5
        self.BG_GMM = GMM()
        self.FG_GMM = GMM()
        self.component_index = np.zeros((self.row, self.col), dtype=np.uint8)

        self.gamma = 0
        self.beta = 0
        self.left_W = np.zeros((self.row, self.col))
        self.upleft_W = np.zeros((self.row, self.col))
        self.up_W = np.zeros((self.row, self.col))
        self.upright_W = np.zeros((self.row, self.col))
        self._init_V()

        self.defi_BG = 0
        self.defi_FG = 1
        self.prob_BG = 2
        self.prob_FG = 3
        self.mask = np.zeros((self.row, self.col), dtype=np.uint8)
        self.alpha = np.zeros((self.row, self.col), dtype=np.uint8)

        self.g = graph(0)

        self.max_iter = 1

    def OriginalIterate(self, p1, p2):
        x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
        self.mask[x1:x2 + 1, y1:y2 + 1] = self.prob_FG
        self.init_with_kmeans()
        for i in range(self.max_iter):
            self.assign_GMM_component()
            self.learn_GMM_parameters()
            self.construct_graph()
            self.estimate_segmentation()

        return self.alpha

    def add_foreground(self, pixels):
        # to be implemented
        return

    def add_background(self, pixels):
        # to be implemented
        return

    def rerun(self):
        # to be implemented
        return np.zeros(self.img_shape, np.uint8)

    def _init_V(self):
        self.gamma = 50
        self.cal_beta()
        self.cal_nearby_W()

    def cal_beta(self):
        left_diff = self.img[:, 1:] - self.img[:, :-1]
        upleft_diff = self.img[1:, 1:] - self.img[:-1, :-1]
        up_diff = self.img[1:, :] - self.img[:-1, :]
        upright_diff = self.img[1:, :-1] - self.img[:-1, 1:]
        self.beta = 0
        self.beta += np.sum(left_diff ** 2)
        self.beta += np.sum(upleft_diff ** 2)
        self.beta += np.sum(up_diff ** 2)
        self.beta += np.sum(upright_diff ** 2)
        self.beta = self.beta / (4 * self.row * self.col - 3 * self.row - 3 * self.col + 2)
        self.beta = 1 / (2 * self.beta)

    def cal_nearby_W(self):
        for i in range(self.row):
            for j in range(self.col):
                color = self.img[i, j]
                if j >= 1:
                    diff = color - self.img[i, j - 1]
                    self.left_W[i, j] = self.gamma * np.exp(-self.beta * np.sum(diff ** 2))
                if i >= 1 and j >= 1:
                    diff = color - self.img[i - 1, j - 1]
                    self.upleft_W[i, j] = self.gamma * np.exp(-self.beta * np.sum(diff ** 2)) / np.sqrt(2)
                if i >= 1:
                    diff = color - self.img[i - 1, j]
                    self.up_W[i, j] = self.gamma * np.exp(-self.beta * np.sum(diff ** 2))
                if i >= 1 and j < self.col - 1:
                    diff = color - self.img[i - 1, j + 1]
                    self.upright_W[i, j] = self.gamma * np.exp(-self.beta * np.sum(diff ** 2)) / np.sqrt(2)

    def init_with_kmeans(self):
        BG_index = np.logical_or(self.mask == self.defi_BG, self.mask == self.prob_BG)
        FG_index = np.logical_or(self.mask == self.defi_FG, self.mask == self.prob_FG)
        BG_pixels = self.img[BG_index]
        FG_pixels = self.img[FG_index]
        BG_KM = kmeans(BG_pixels, 3, self.k, 20)
        FG_KM = kmeans(FG_pixels, 3, self.k, 20)
        BG_KM.run()
        FG_KM.run()
        BG_by_component = BG_KM.output()
        FG_by_component = FG_KM.output()
        for i in range(self.k):
            for j in BG_by_component[i]:
                self.BG_GMM.add_pixel(j, i)
            for j in FG_by_component[i]:
                self.FG_GMM.add_pixel(j, i)
        self.BG_GMM.update()
        self.FG_GMM.update()
        self.BG_GMM.clear()
        self.FG_GMM.clear()

    def assign_GMM_component(self):
        for i in range(self.row):
            for j in range(self.col):
                if self.mask[i, j] == self.defi_BG or self.mask[i, j] == self.prob_BG:
                    self.component_index[i, j] = self.BG_GMM.most_likely_pixel_component(self.img[i, j])
                else:
                    self.component_index[i, j] = self.FG_GMM.most_likely_pixel_component(self.img[i, j])

    def learn_GMM_parameters(self):
        for i in range(self.k):
            BG_index = np.logical_and(self.component_index == i,
                                      np.logical_or(self.mask == self.defi_BG, self.mask == self.prob_BG))
            FG_index = np.logical_and(self.component_index == i,
                                      np.logical_or(self.mask == self.defi_FG, self.mask == self.prob_FG))
            BG_pixel = self.img[BG_index]
            FG_pixel = self.img[FG_index]
            for j in BG_pixel:
                self.BG_GMM.add_pixel(j, i)
            for j in FG_pixel:
                self.FG_GMM.add_pixel(j, i)
        self.BG_GMM.update()
        self.FG_GMM.update()
        self.BG_GMM.clear()
        self.FG_GMM.clear()

    def construct_graph(self):
        self.g = graph(self.row * self.col + 2)
        self.b = np.zeros((self.row, self.col))
        self.f = np.zeros((self.row, self.col))
        for i in range(self.row):
            for j in range(self.col):
                a = i * self.col + j + 1
                t = self.row * self.col + 1
                w1 = -np.log(self.BG_GMM.prob_pixel_GMM(self.img[i, j])).astype(np.int16)
                self.b[i, j] = w1
                self.g.addedge(0, a, w1)
                self.g.addedge(a, 0, 0)
                w2 = -np.log(self.FG_GMM.prob_pixel_GMM(self.img[i, j])).astype(np.int16)
                self.f[i, j] = w2
                self.g.addedge(a, t, w2)
                self.g.addedge(t, a, 0)
                if j >= 1:
                    b = i * self.col + j
                    self.g.addedge(a, b, self.left_W[i, j].astype(np.int8))
                    self.g.addedge(b, a, self.left_W[i, j].astype(np.int8))
                if i >= 1 and j >= 1:
                    b = (i - 1) * self.col + j
                    self.g.addedge(a, b, self.upleft_W[i, j].astype(np.int8))
                    self.g.addedge(b, a, self.upleft_W[i, j].astype(np.int8))
                if i >= 1:
                    b = (i - 1) * self.col + j + 1
                    self.g.addedge(a, b, self.up_W[i, j].astype(np.int8))
                    self.g.addedge(b, a, self.up_W[i, j].astype(np.int8))
                if i >= 1 and j < self.col - 1:
                    b = (i - 1) * self.col + j + 2
                    self.g.addedge(a, b, self.upright_W[i, j].astype(np.int8))
                    self.g.addedge(b, a, self.upright_W[i, j].astype(np.int8))

    def estimate_segmentation(self):
        self.g.dinic()
        e = self.g.head[0]
        while e != -1:
            t = self.g.edges[e].to
            i = (t - 1) // self.col
            j = (t - 1) % self.col
            if self.g.edges[e].w <= 0:
                if self.mask[i, j] == self.prob_FG:
                    self.mask[i, j] = self.prob_BG
            else:
                if self.mask[i, j] == self.prob_BG:
                    self.mask[i, j] = self.prob_FG
            e = self.g.edges[e].nex
        self.alpha = np.zeros(self.img_shape)
        FG_index = np.logical_or(self.mask == self.prob_FG, self.mask == self.defi_FG)
        self.alpha[FG_index] = 1


class edge:
    def __init__(self, a, b, c):
        self.to = a
        self.w = b
        self.nex = c


class graph:
    def __init__(self, vertex_num):
        self.head = [-1 for i in range(vertex_num)]
        self.dis = [-1 for i in range(vertex_num)]
        self.cur = [0 for i in range(vertex_num)]
        self.edges = []
        self.n = vertex_num
        self.flow = 0
        self.m = 0

    def addedge(self, source, destination, weight):
        e = edge(destination, weight, self.head[source])
        self.head[source] = self.m
        self.m += 1
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
        c_e = self.cur[u]
        while c_e != -1 and limit > 0:
            _to = self.edges[c_e].to
            if self.dis[_to] == self.dis[u] + 1 and self.edges[c_e].w > 0 and limit > 0:
                flow2 = self.dfs(_to, min(limit, self.edges[c_e].w))
                flow1 += flow2
                self.edges[c_e].w -= flow2
                self.edges[c_e ^ 1].w += flow2
                if self.edges[c_e].w > 0:
                    self.cur[u] = c_e
                limit -= flow2
            c_e = self.edges[c_e].nex
        if flow1 == 0:
            self.dis[u] = -1
        return flow1

    # 默认第一个点是源点，最后一个点是汇点
    def dinic(self):
        self.flow = 0
        cnt = 0
        while self.bfs():
            self.cur = self.head.copy()
            self.flow += self.dfs(0, 100000000000000)
        return self.flow

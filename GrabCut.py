import cv2
import numpy as np
import ctypes


class kmeans:
    def __init__(self, data, dim, n, max_iter):
        self.row = int(data.size / dim)
        self.dim = dim
        self.data = data.reshape(self.row, dim).copy().astype(np.double)
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


class Pointer:
    def __init__(self, var):
        self.id = id(var)

    def get_value(self):
        return ctypes.cast(self.id, ctypes.py_object).value


class Vertex:
    def __init__(self):
        self.next = 0  # next vertex
        self.parent = 0  # edge to parent
        self.first = 0  # first edge
        self.ts = 0  # to source
        self.dist = 0  # dist to root
        self.weight = 0
        self.t = 0  # tree


class Edge:
    def __init__(self):
        self.dst = 0
        self.next = 0
        self.weight = 0.0


class GCGraph:
    def __init__(self, vertex_count, edge_count):
        self.vertex_count = vertex_count
        self.edge_count = edge_count
        self.vertexs = []
        self.edges = []
        self.flow = 0

    def add_vertex(self):
        v = Vertex()
        self.vertexs.append(v)
        return len(self.vertexs) - 1

    def add_edges(self, i, j, w, revw):
        a = len(self.edges)

        fromI = Edge()
        fromI.dst = j
        fromI.next = self.vertexs[i].first
        fromI.weight = w
        self.vertexs[i].first = a
        self.edges.append(fromI)

        toI = Edge()
        toI.dst = i
        toI.next = self.vertexs[j].first
        toI.weight = revw
        self.vertexs[j].first = a + 1
        self.edges.append(toI)

    def add_term_weights(self, i, source_weight, sink_weight):
        dw = self.vertexs[i].weight
        if dw > 0:
            source_weight += dw
        else:
            sink_weight -= dw
        if source_weight < sink_weight:
            self.flow += source_weight
        else:
            self.flow += sink_weight
        self.vertexs[i].weight = source_weight - sink_weight

    def max_flow(self):
        TERMINAL = -1
        ORPHAN = -2
        stub = Vertex()
        nilNode = Pointer(stub)
        first = Pointer(stub)
        last = Pointer(stub)
        curr_ts = 0
        stub.next = nilNode.get_value()
        orphans = []

        for i in range(len(self.vertexs)):
            v = self.vertexs[i]
            v.ts = 0
            if v.weight != 0:
                last.get_value().next = v
                last.id = id(v)
                v.dist = 1
                v.parent = TERMINAL
                v.t = v.weight < 0
            else:
                v.parent = 0

        first.id = id(first.get_value().next)
        last.get_value().next = nilNode.get_value()
        nilNode.get_value().next = 0
        while True:
            e0 = -1
            while first.get_value() != nilNode.get_value():
                v = first.get_value()
                if v.parent:
                    vt = v.t
                    ei = v.first
                    while ei != 0:
                        if self.edges[ei ^ vt].weight == 0:
                            ei = self.edges[ei].next
                            continue
                        u = self.vertexs[self.edges[ei].dst]
                        if not u.parent:
                            u.t = vt
                            u.parent = ei ^ 1
                            u.ts = v.ts
                            u.dist = v.dist + 1
                            if not u.next:
                                u.next = nilNode.get_value()
                                last.get_value().next = u
                                last.id = id(u)
                            ei = self.edges[ei].next
                            continue
                        if u.t != vt:
                            e0 = ei ^ vt
                            break
                        if u.dist > v.dist + 1 and u.ts <= v.ts:
                            u.parent = ei ^ 1
                            u.ts = v.ts
                            u.dist = v.dist + 1
                        ei = self.edges[ei].next
                    if e0 > 0:
                        break
                first.id = id(first.get_value().next)
                v.next = 0
            if e0 <= 0:
                break

            minWeight = self.edges[e0].weight
            for k in range(1, -1, -1):
                v = self.vertexs[self.edges[e0 ^ k].dst]
                while True:
                    ei = v.parent
                    if ei < 0:
                        break
                    weight = self.edges[ei ^ k].weight
                    minWeight = min(minWeight, weight)
                    v = self.vertexs[self.edges[ei].dst]
                weight = abs(v.weight)
                minWeight = min(minWeight, weight)

            self.edges[e0].weight -= minWeight
            self.edges[e0 ^ 1].weight += minWeight
            self.flow += minWeight

            for k in range(1, -1, -1):
                v = self.vertexs[self.edges[e0 ^ k].dst]
                while True:
                    ei = v.parent
                    if ei < 0:
                        break
                    self.edges[ei ^ (k ^ 1)].weight += minWeight
                    self.edges[ei ^ k].weight -= minWeight
                    if self.edges[ei ^ k].weight == 0:
                        orphans.append(v)
                        v.parent = ORPHAN
                    v = self.vertexs[self.edges[ei].dst]
                v.weight = v.weight + minWeight * (1 - k * 2)
                if v.weight == 0:
                    orphans.append(v)
                    v.parent = ORPHAN
            curr_ts += 1

            while len(orphans) != 0:
                v2 = orphans.pop()
                minDist = float('inf')
                e0 = 0
                vt = v2.t

                ei = v2.first
                bcount = 0
                while ei != 0:
                    bcount += 1
                    if self.edges[ei ^ (vt ^ 1)].weight == 0:
                        ei = self.edges[ei].next
                        continue
                    u = self.vertexs[self.edges[ei].dst]
                    if u.t != vt or u.parent == 0:
                        ei = self.edges[ei].next
                        continue

                    d = 0
                    while True:
                        if u.ts == curr_ts:
                            d += u.dist
                            break
                        ej = u.parent
                        d += 1
                        # print(d)
                        if ej < 0:
                            if ej == ORPHAN:
                                d = float('inf') - 1
                            else:
                                u.ts = curr_ts
                                u.dist = 1
                            break
                        u = self.vertexs[self.edges[ej].dst]
                    d += 1
                    if d < float("inf"):
                        if d < minDist:
                            minDist = d
                            e0 = ei
                        u = self.vertexs[self.edges[ei].dst]
                        while u.ts != curr_ts:
                            u.ts = curr_ts
                            d -= 1
                            u.dist = d
                            u = self.vertexs[self.edges[u.parent].dst]

                    ei = self.edges[ei].next
                v2.parent = e0
                if v2.parent > 0:
                    v2.ts = curr_ts
                    v2.dist = minDist
                    continue

                v2.ts = 0
                ei = v2.first
                while ei != 0:
                    u = self.vertexs[self.edges[ei].dst]
                    ej = u.parent
                    if u.t != vt or (not ej):
                        ei = self.edges[ei].next
                        continue
                    if self.edges[ei ^ (vt ^ 1)].weight and (not u.next):
                        u.next = nilNode.get_value()
                        last.get_value().next = u
                        last.id = id(u)
                    if ej > 0 and self.vertexs[self.edges[ej].dst] == v2:
                        orphans.append(u)
                        u.parent = ORPHAN
                    ei = self.edges[ei].next
        return self.flow

    def _init_active_set(self):
        for i in range(self.vertex_count):
            v = self.vertexs[i]
            v.ts = 0
            if v.weight == 0:
                v.parent = -3
            else:
                self.last.next = v
                self.last = v
                v.parent = self.terminal
                v.t = v.weight < 0
                self.dist = 1
        self.first = self.first.next
        self.last.next = self.nilNode
        self.nilNode.next = 0

    def search(self):
        e0 = -3
        while self.first != self.nilNode:
            v = self.first
            ei = v.first
            while ei != -3:
                if self.edges[ei ^ v.t] == 0:
                    ei = self.edges[ei].next
                    continue
                u = self.vertexs[self.edges[ei].dst]
                if u.parent == -3:
                    u.t = v.t
                    u.dist = v.dist + 1
                    u.parent = ei ^ 1
                    u.ts = v.ts
                    if u.next == 0:
                        u.next = self.nilNode
                        self.last.next = u
                        self.last = u
                if u.t != v.t:
                    e0 = ei ^ v.t
                    break
                if u.dist > v.dist + 1 and u.ts <= v.ts:
                    u.parent = ei ^ 1
                    u.ts = v.ts
                    u.dist = v.dist + 1
                ei = self.edges[ei].next
            if e0 >= 0:
                break
            self.first = self.first.next
            v.next = 0
        return e0

    def augment(self, e0):
        minweight = self.edges[e0].weight
        for k in [1, 0]:
            v = self.vertexs[self.edges[e0 ^ k].dst]
            while True:
                ei = v.parent
                if ei < 0:
                    break
                minweight = min(minweight, self.edges[ei ^ k].weight)
                v = self.vertexs[self.edges[ei].dst]
            minweight = min(minweight, abs(v.weight))

        self.edges[e0].weight -= minweight
        self.edges[e0 ^ 1].weight += minweight
        self.flow += minweight
        for k in [1, 0]:
            v = self.vertexs[self.edges[e0 ^ k].dst]
            while True:
                ei = v.parent
                if ei < 0:
                    break
                self.edges[ei ^ k].weight -= minweight
                self.edges[ei ^ k ^ 1].weight += minweight
                if self.edges[ei ^ k].weight == 0:
                    self.orphans.append(v)
                    v.parent = self.orphan
                v = self.vertexs[self.edges[ei].dst]
            v.weight += minweight * (1 - 2 * k)
            if v.weight == 0:
                self.orphans.append(v)
                v.parent = self.orphan

    def adoption(self):
        while len(self.orphans) != 0:
            v = self.orphans.pop()
            e0 = -3
            ei = v.first
            mindist = float('inf')
            while ei != -3:
                u = self.vertexs[self.edges[ei].dst]
                if self.edges[ei ^ 1 ^ v.t].weight == 0 or u.t != v.t or u.parent == -3:
                    ei = self.edges[ei].next
                    continue
                d = 1
                while True:
                    if u.ts == self.curr_ts:
                        d += u.dist
                        break
                    ej = u.parent
                    d += 1
                    if ej < 0:
                        if ej == self.orphan or ej == -3:
                            d = float('inf')
                        else:
                            u.dist = 1
                            u.ts = self.curr_ts
                        break
                    u = self.vertexs[self.edges[ej].dst]
                if d < float('inf'):
                    if d < mindist:
                        mindist = d
                        e0 = ei
                    while u.ts != self.curr_ts:
                        d -= 1
                        u.ts = self.curr_ts
                        u.dist = d
                        u = self.vertexs[self.edges[u.parent].dst]
                ei = self.edges[ei].next

            v.parent = e0
            if v.parent > 0:
                v.ts = self.curr_ts
                v.dist = mindist
                continue

            v.ts = 0
            ei = v.first
            while ei != -3:
                u = self.vertexs[self.edges[ei].dst]
                ej = u.parent
                if ej == -3 or u.t != v.t:
                    ei = self.edges[ei].next
                    continue
                if u.next == 0 and self.edges[ei ^ v.t ^ 1].weight > 0:
                    u.next = self.nilNode
                    self.last.next = u
                    self.last = u
                if ej >= 0 and self.vertexs[self.edges[ej].dst] == v:
                    self.orphans.append(u)
                    u.parent = self.orphan
                ei = self.edges[ei].next

    def insource_segment(self, i):
        return self.vertexs[i].t == 0


class GCEngine:
    def __init__(self, img):
        self.img = img.astype(np.double)
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

        self.max_iter = 2

    def OriginalIterate(self, p1, p2):
        x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
        self.mask[x1:x2 + 1, y1:y2 + 1] = self.prob_FG
        self.init_with_kmeans()
        for i in range(self.max_iter):
            self.assign_GMM_component()
            self.learn_GMM_parameters()
            self.construct_gcgraph()
            self.estimate_segmentation()

        return self.alpha

    def add_foreground(self, pixels):
        self.mask[pixels == 1] = self.defi_FG
        return

    def add_background(self, pixels):
        self.mask[pixels == 1] = self.defi_BG
        return

    def rerun(self):
        self.assign_GMM_component()
        self.learn_GMM_parameters()
        self.construct_gcgraph()
        self.estimate_segmentation()
        return self.alpha

    def run(self, trimap):
        self.mask[trimap == 0] = self.defi_BG
        self.mask[trimap == 1] = self.prob_FG
        self.mask[trimap == 2] = self.defi_FG
        self.init_with_kmeans()
        for i in range(self.max_iter):
            self.assign_GMM_component()
            self.learn_GMM_parameters()
            self.construct_gcgraph()
            self.estimate_segmentation()
        return self.alpha

    def _init_V(self):
        self.gamma = 30
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
        BG_KM = kmeans(BG_pixels, 3, self.k, 10)
        FG_KM = kmeans(FG_pixels, 3, self.k, 10)
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

    def construct_gcgraph(self, lam=450):
        vertex_count = self.col * self.row
        edge_count = 2 * (4 * vertex_count - 3 * (self.row + self.col) + 2)
        self.graph = GCGraph(vertex_count, edge_count)
        self.b = np.zeros((self.row, self.col))
        self.f = np.zeros((self.row, self.col))
        for y in range(self.row):
            for x in range(self.col):
                vertex_index = self.graph.add_vertex()
                color = self.img[y, x]
                if self.mask[y, x] == self.prob_BG or self.mask[y, x] == self.prob_FG:
                    fromSource = -np.log(self.BG_GMM.prob_pixel_GMM(color)+0.0001)
                    toSink = -np.log(self.FG_GMM.prob_pixel_GMM(color)+0.0001)
                elif self.mask[y, x] == self.defi_BG:
                    fromSource = 0
                    toSink = lam
                else:
                    fromSource = lam
                    toSink = 0
                self.b[y, x] = fromSource
                self.f[y, x] = toSink
                self.graph.add_term_weights(vertex_index, fromSource, toSink)
                if x > 0:
                    w = self.left_W[y, x]
                    self.graph.add_edges(vertex_index, vertex_index - 1, w, w)
                if x > 0 and y > 0:
                    w = self.upleft_W[y, x]
                    self.graph.add_edges(vertex_index, vertex_index - self.col - 1, w, w)
                if y > 0:
                    w = self.up_W[y, x]
                    self.graph.add_edges(vertex_index, vertex_index - self.col, w, w)
                if x < self.col - 1 and y > 0:
                    w = self.upright_W[y, x]
                    self.graph.add_edges(vertex_index, vertex_index - self.col + 1, w, w)

    def estimate_segmentation(self):
        a = self.graph.max_flow()
        for y in range(self.row):
            for x in range(self.col):
                if self.mask[y, x] == self.prob_BG or self.mask[y, x] == self.prob_FG:
                    if self.graph.insource_segment(y * self.col + x):  # Vertex Index
                        self.mask[y, x] = self.prob_FG
                    else:
                        self.mask[y, x] = self.prob_BG
        self.alpha = np.zeros((self.row, self.col))
        FG_index = np.logical_or(self.mask == self.prob_FG, self.mask == self.defi_FG)
        self.alpha[FG_index] = 1

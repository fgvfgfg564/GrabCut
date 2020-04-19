# 后端文件

import cv2
import numpy as np


class GCEngine:
    def __init__(self, img):
        # to be implemented
        self.img = img
        self.img_size = img.shape

    def original_iterate(self, p1, p2):
        # to be implemented
        self.x1, self.y1, self.x2, self.y2 = p1[0], p1[1], p2[0], p2[1]
        self.alpha = np.zeros((self.img_shape[0], self.img_shape[1]))
        self.alpha[self.x1:self.x2 + 1, self.y1:self.y2 + 1] = 1
        self.init_GMM()
        max_iteration = 10
        for i in range(max_iteration):
            self.assign_k()
            self.update_GMM()
            self.graphcut()  # to be implemented

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
        #GMM parameters
        self.k=np.zeros_like(self.alpha)
        self.K=5
        self.p=np.zeros((2,self.K))
        self.mu=np.zeros((2,self.K,3))
        self.Siginv=np.zeros((2,self.K,3,3))
        self.sig=np.ones((2,self.K))
        self.p[:]=1/(math.sqrt(2*math.pi)**3)
        self.Siginv[:,:]=np.array([[1,0,0],[0,1,0],[0,0,1]])
        print("f",self.Siginv)
        #divide object and background pixels
        mask=np.array(self.alpha==0)
        A=self.img[mask]#object pixels
        B=self.img[~mask]#background pixels

        #initialize with k-means
        index1 = np.random.choice(range(A.shape[0]), self.K, replace=False)
        index2 = np.random.choice(range(B.shape[0]), self.K, replace=False)
        self.mu[0]=A[index1]
        self.mu[1]=B[index2]
        A_uniq=np.unique(A,axis=0)
        B_uniq=np.unique(B,axis=0)
        if A_uniq.shape[0]>self.K:
            index1 = np.random.choice(range(A_uniq.shape[0]), self.K, replace=False)
            self.mu[0] = A_uniq[index1]
        if B_uniq.shape[0]>self.K:
            index2 = np.random.choice(range(B_uniq.shape[0]), self.K, replace=False)
            self.mu[1] = B_uniq[index2]

        #begin iteration
        max_iteration=10
        for i in range(max_iteration):
            A_group=[[],[],[],[],[]]
            B_group=[[],[],[],[],[]]

            #group
            for j in range(A.shape[0]):
                distance=np.sum((A[j]-self.mu[0])**2,axis=1)
                type=np.argmin(distance)
                A_group[type].append(A[j])
            for j in range(B.shape[0]):
                distance = np.sum((B[j] - self.mu[1]) ** 2, axis=1)
                type = np.argmin(distance)
                B_group[type].append(B[j])

            self.update(A_group,B_group)

    #A_group is a 2-dimensional list of the pixels in the background,B_group is a 2-dimensional list of the grouped pixels in the object
    def update(self,A_group,B_group):
        for j in range(self.K):
            n1 = np.array(A_group[j])
            n2 = np.array(B_group[j])
            if n1.shape[0] != 0:
                d1 = math.fabs(np.linalg.det(n1.T @ n1 / n1.shape[0]))
                if d1 > 0:
                    self.mu[0][j] = np.mean(n1, axis=0)
                    self.Siginv[0][j] = np.linalg.inv(n1.T @ n1 / n1.shape[0])
                    self.sig[0][j] = math.sqrt(d1)
                    self.p[0][j] = 1 / (math.sqrt(d1) * (math.sqrt(2 * math.pi) ** 3))
            if n2.shape[0] != 0:
                d2 = math.fabs(np.linalg.det(n2.T @ n2 / n2.shape[0]))
                if d2 > 0:
                    self.mu[1][j] = np.mean(n2, axis=0)
                    self.Siginv[1][j] = np.linalg.inv(n2.T @ n2 / n2.shape[0])
                    self.sig[1][j] = math.sqrt(d2)
                    self.p[1][j] = 1 / (math.sqrt(d2) * (math.sqrt(2 * math.pi) ** 3))

    def assign_k(self):
        for i in range(self.img_shape[0]):
            for j in range(self.img_shape[1]):
                if self.alpha[i,j]==0:
                    P=self.p[0]*np.exp(-0.5*self.img[i,j]@self.Siginv[0]@self.img[i,j].reshape(-1,1).reshape(-1))
                    self.k[i,j]=np.argmin(P)
                else:
                    P=self.p[1]*np.exp(-0.5*self.img[i,j]@self.Siginv[1]@self.img[i,j].reshape(-1,1).reshape(-1))
                    self.k[i, j] = np.argmin(P)

    def update_GMM(self):
        A_group = [[], [], [], [], []]
        B_group = [[], [], [], [], []]
        for i in range(self.img_shape[0]):
            for j in range(self.img_shape[1]):
                if self.alpha[i,j]==0:
                    A_group[self.k[i,j]].append(self.img[i,j])
                else:
                    B_group[self.k[i,j]].append(self.img[i,j])
        self.update(A_group,B_group)

    def graphcut(self):
        return self.alpha
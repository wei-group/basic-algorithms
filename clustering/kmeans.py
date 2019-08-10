# 2019-08-07
# author: Ryb
# 尝试自己实现kmeans聚类算法

# kmeans算法
    #  1. 初始化k个族中心
    #  2. 计算每个样本与k个族中心距离，标记样本为距离最短的族
    #  3. 重新确定K族中心(族中平均位置)
    #  4. 循环2-3，直到前后两次所有族中心距离变化<eps

"""
# 输入：
    X: 2d数组, 形如(n_samples, m_features)，n_samples表示样本数，m_features表示特征维数
    K: int, 超参数，指定族格式
    metric：str, 距离类型，默认为欧式距离'Euler',其他暂时为实现
    eps: float, 精度（当族中心位置更新变化<eps时停止)
    random_state: 随机种子    
# 输出：
    centers: K族中心向量, 2d数组, 形如(K, m_features)
    pred: 1-d数组,长度为n_samples 
"""

import numpy as np
import random   # 用python的random模块，不用numpy的random

class kmeans: # 创建kmeans类
    
    # 初始化函数
    def __init__(self, X=None, K=2, metric='Euler', eps=1e-6, init_centers=None, random_state=None):
        self.X = X
        self.K = K
        self.metric = metric
        self.eps = eps      
        self.centers = init_centers
        self.random_state = random_state
        # if not self.centers and not self.X:
        #     if random_state is not None:
        #         random.seed(random_state)              
        #     idx = random.sample(range(self.X.shape[0]), self.K)
        #     self.centers = self.X[idx,:]
    
    
    # 距离函数
    def calc_dist(self, x, c):  
        """
        # 如果主样本与单中心计算欧式距离，返回 np.sqrt(np.power(x-c,2)).sum() 即可；
        # 考虑到扩展其他距离计算方式，采用用闵可夫斯基距离，当lp=2时候即为欧式距离
        # 单样本-单中心的距离计算，返回dist.sum()
        # 单样本-多中心的距离计算，返回dist.sum(axis=1)
        """     
        if self.metric=='Euler':   
            lp = 2   
        dist = np.power(np.power(x-c,lp), 1/lp) 
        if len(dist.shape)==1: 
            return dist.sum() # 单样本，单中心
        else: 
            return dist.sum(axis=1) # 单样本，多中心


    # 迭代(训练)
    def fit(self, X):
        
        # 样本
        if X is not None:
            self.X = X

        # 样本形状    
        n_samples, n_features = self.X.shape  

        # 设置随机种子
        if self.random_state is not None:
                random.seed(self.random_state)    

                  
        # 初始化聚类中心
        if self.centers is None:  
            """
            # idx = np.random.randint(low=0, hight=n_sample,size=self.K)
            # 用randint初始化，有重复；重复的族中心，会导致族中分配不到成员，求均值NaN
            # 更新的族中心后，中心向量NaN 
            #       
            """
            idx = idx = random.sample(range(n_samples), self.K)
            self.centers = X[idx,:]
        
        
        # 初始样本的族标记-1
        pred = np.array([-1]*n_samples)
        
        iter = 0
        stop = False # 结束标志
        while (not stop):
            iter +=1
            print(iter)
            # 遍历所有样本，划分族
            # for i in range(n_samples):
            #     min_dist = np.inf
            #     c = -1
            #     # 遍历所有族中心向量
            #     for k in range(self.K):
            #         dist = self.calc_dist(X[i,:], self.centers[k,:])
            #         if dist < min_dist:
            #             min_dist = dist
            #             c = k
            #     pred[i] =c

            for i in range(n_samples):                             
                dists = self.calc_dist(X[i,:], self.centers)
                pred[i] = np.argmin(dists)
        
            
            # 重新确定族中心
            new_centers = np.zeros((self.K, n_features))           
            for k in range(self.K):                
                new_centers[k,:] = X[pred==k,:].mean(axis=0)

            # 判断停止条件
            delta = abs(new_centers - self.centers)
            flg = delta <self.eps
            stop = flg.all()
            self.centers  = new_centers

        return pred, self.centers

    # 族预测
    def predict(self, X):
         # 遍历所有样本，划分族
        pred = np.array([-1]*n_samples)
        for i in range(n_samples):                             
            dists = self.calc_dist(X[i,:], self.centers)
            pred[i] = np.argmin(dists)
        return pred

if __name__ == "__main__":
         
    import matplotlib.pyplot as plt 
    from sklearn.datasets import make_blobs

    # 生成数据
    n_samples = 1500
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)  
    
    # 调用kmeans
    model = kmeans(K=3, eps=1e-3, random_state=1) 
    pred, centers = model.fit(X)
    n_samples, _ = X.shape
    
    # 族预测，如果仅是训练数据，直接用fit(X)返回的族划分
    # pred = model.predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=pred)
    plt.title("kmeans")    
    plt.show()










            




            
                    


        


    



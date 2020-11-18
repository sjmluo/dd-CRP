import numpy as np
from scipy.special import gammaln
from scipy.special import logsumexp

class distance_depdendent_chinese_resturant_process:
    def __init__(self, X, lhood_fn, distance, delay):
        """
        Parameters:
        -----------
        X: np.ndarray
            The obersvations. Where each row represents an observation.
        lhood_fn: function
            The likelihood function for ddCRP with single input X
        distance: function
            The distance function with input two inputs idex i and j
        delay: function
            Function to apply a window to the distance function with single input X
            
        Returns:
        --------
        None
        """
        self.X = X
        self.N = len(self.X)
        self.lhood_fn = lhood_fn
        self.distance = distance
        self.delay = delay
        
        self.cluster = np.array([0]*self.N)
        self.link = np.array([0]*self.N)
        self.prior = np.random.random(self.N*self.N).reshape((self.N,self.N))
        self.merged_lhood = np.random.random(self.N)
    
    def _get_linked(self, i):
        """
        get the customers linked to customer i
        
        Parameters:
        -----------
        i: int
            Customer i
        
        Returns:
        --------
        c: list
            All customers linked to customer i
        """
        c = list()
        q = list()
        q.append(i)
        while q:
            cur = q[0]
            c.append(cur)
            for k in range(0,len(self.link)):
                if (self.link[k] == cur) and (k not in c) and (k not in q):
                    q.append(k)
            q = q[1:]
        return c

    def train(self, n_iter, alpha=0.2, verbose=False, verbose_step=1, remove_empty_index=False):
        """
        Train the model.
        
        Parameters:
        -----------
        n_iter: int
            Number of iterations
        alpha: float
            The concentration parameter
        verbose: bool
            Printing flag
        verbose_step: int
            verbose printing step
        remove_empty_index:
            Remove empty table index

		Returns:
		--------
		None
        """
        lhood = np.array([self.lhood_fn(self.X[np.where(self.cluster == x)]) for x in self.cluster])

        #prior of each customer
        for i in range(self.N):
            for j in range(self.N):
                if i==j:
                    self.prior[i][j] = np.log(alpha)
                else:
                    if self.delay(self.distance(i,j)) != 0:
                        self.prior[i][j] = np.log(self.delay(self.distance(i,j)))
                    else:
                        self.prior[i][j] = -np.inf

        for t in range(n_iter):
            obs_lhood = 0
            for i in range(self.N):
                #remove the ith's link
                old_link = self.link[i]
                old_cluster = self.cluster[old_link]
                self.cluster[i] = i
                self.link[i] = i
                linked = self._get_linked(i)
                self.cluster[linked] = i

                if old_cluster not in linked :
                    idx = i
                    lhood[old_cluster] = self.lhood_fn(self.X[idx])
                    lhood[i] = self.lhood_fn(self.X[linked])


                #calculate the likelihood of the merged cluster 
                for j in np.unique(self.cluster):

                    if j == self.cluster[i] :
                        self.merged_lhood[j] = 2*self.lhood_fn(self.X[linked])
                    else: 
                        self.merged_lhood[j] = self.lhood_fn(np.concatenate((self.X[linked] , self.X[np.where(self.cluster == j)])))

                log_prob = np.array([self.prior[i][x] + self.merged_lhood[self.cluster[x]] - lhood[self.cluster[x]]-lhood[self.cluster[i]] for x in np.arange(self.N)])
                prob = np.exp(log_prob - logsumexp(log_prob))

                #sample z_i
                self.link[i] = np.random.choice(np.arange(self.N),1,p=prob)

                #update the likelihood if the link sample merge two cluster
                new_cluster = self.cluster[self.link[i]]
                if new_cluster != i:
                    self.cluster[linked] = new_cluster
                    lhood[new_cluster] = self.merged_lhood[new_cluster]

                #cal the likelihood of all obs
                for u in np.unique(self.cluster):
                    obs_lhood = obs_lhood + lhood[u]
            
            if verbose and ((t % verbose_step) == 0):
                print("iter:", t)
                print("cluster:\n", self.cluster)
                print("link:\n", self.link)
                print("likelihood:", obs_lhood)

        if remove_empty_index:
            old_cluster = np.array(self.cluster)
            old_link = np.array(self.link)
            for cn, co in enumerate(np.unique(old_cluster)):
                self.cluster[old_cluster == co] = cn
                self.link[old_link == co] = cn
    
    def get_cluster(self):
        return self.cluster
    
    def get_link(self):
        return self.link

if __name__ == "__main__":
    def dirichlet_likelihood(Xp, hyper):
        """
        Parameters:
        -----------
        Xp: numpy.ndarray
            Each observation of X
        hyper: float
            The hyper-parameter for the Dirichlet distribution
        
        Returns:
        lh: float
            The likelihood
        """
        if len(Xp.shape) == 2: 
            X =sum(Xp)
        else:
            X = Xp
        idx = np.where(X!=0)
        lh = gammaln(len(X)*hyper) + sum(gammaln(X[idx]+hyper))\
        -len(idx)*gammaln(hyper)  - gammaln(sum(X)+len(X) * hyper)
        return lh

    def linear_distance(i,j):
        """
        Simple distance between i and j
        """
        return i-j

    def window_delay(a,size=1):
        """
        Simple window delay for a
        """
        if abs(a) <= size and a >= 0:
            return 1;
        else:
            return 0;
    
    #a demo
    obs = np.array([[10,0,0,0,0],
                    [10,0,0,0,0],
                    [5,0,0,0,0],
                    [11,0,0,0,1],
                    [0,10,0,0,1],
                    [0,10,0,0,0],
                    [0,0,10,0,0],
                    [0,1,10,0,0],
                    [20,0,2,0,0],
                    [10,0,0,1,0],
                    [10,1,0,10,0],
                    [10,0,2,10,0],
                    [10,0,0,10,0],
                    [10,1,0,1,0],
                    [10,0,0,0,0]])

    #initalize parameters
    n_iter =200
    hyper = 0.01
    alpha = 0.2
    window_size = 20

    #bookkeeper data
    #cluster,link,obs,lhood,
    lhood_fn = lambda x:dirichlet_likelihood(x,hyper)
    distance = linear_distance
    delay = lambda x:window_delay(x,window_size)
    ddCRP = distance_depdendent_chinese_resturant_process(obs,lhood_fn,distance,delay)
    ddCRP.train(n_iter,alpha,verbose=False)
    cluster = ddCRP.get_cluster()
    link = ddCRP.get_link()
    
    for c in np.unique(cluster):
        print('Cluster:', c)
        print(obs[cluster == c])
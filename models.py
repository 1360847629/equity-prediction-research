"""Small NumPy-only reference estimators; no sklearn/SciPy dependency."""
import numpy as np

class Ridge:
    def __init__(self,alpha=100): self.alpha=alpha
    def fit(self,X,y):
        X=np.asarray(X,dtype=float);y=np.asarray(y,dtype=float)
        self.mean=X.mean(axis=0);self.scale=X.std(axis=0);self.scale[self.scale==0]=1
        z=(X-self.mean)/self.scale;self.intercept=y.mean()
        self.coef=np.linalg.solve(z.T@z+self.alpha*np.eye(z.shape[1]),z.T@(y-self.intercept))
        return self
    def predict(self,X):return self.intercept+((np.asarray(X)-self.mean)/self.scale)@self.coef

class HistGradientBoostingRegressor:
    """Squared-error gradient boosting with quantile-binned decision stumps."""
    def __init__(self,max_iter=100,max_leaf_nodes=15,learning_rate=.08,l2_regularization=10,random_state=42):
        self.rounds=max_iter;self.bins=max_leaf_nodes;self.rate=learning_rate;self.l2=l2_regularization
    def fit(self,X,y):
        X=np.asarray(X,dtype=float);y=np.asarray(y,dtype=float)
        self.edges=[np.unique(np.quantile(X[:,j],np.linspace(0,1,self.bins+1)[1:-1])) for j in range(X.shape[1])]
        codes=[np.searchsorted(e,X[:,j]) for j,e in enumerate(self.edges)]
        self.base=y.mean();pred=np.full(len(y),self.base);self.trees=[]
        for _ in range(self.rounds):
            residual=y-pred;best=None
            for j,c in enumerate(codes):
                n=np.bincount(c,minlength=len(self.edges[j])+1);s=np.bincount(c,weights=residual,minlength=len(n))
                left_n=np.cumsum(n)[:-1];left_s=np.cumsum(s)[:-1]
                right_n=len(y)-left_n;right_s=residual.sum()-left_s
                gains=left_s**2/(left_n+self.l2)+right_s**2/(right_n+self.l2)
                gains[(left_n<100)|(right_n<100)]=-np.inf
                k=int(np.argmax(gains))
                if best is None or gains[k]>best[0]:best=(gains[k],j,k,left_s[k]/(left_n[k]+self.l2),right_s[k]/(right_n[k]+self.l2))
            _,j,k,left,right=best
            self.trees.append((j,self.edges[j][k],left,right))
            pred+=self.rate*np.where(codes[j]<=k,left,right)
        return self
    def predict(self,X):
        X=np.asarray(X,dtype=float);p=np.full(len(X),self.base)
        for j,threshold,left,right in self.trees:p+=self.rate*np.where(X[:,j]<=threshold,left,right)
        return p

def mean_absolute_error(y,p):return np.mean(np.abs(np.asarray(y)-p))
def mean_squared_error(y,p):return np.mean((np.asarray(y)-p)**2)

class NeuralRegressor:
    """One-hidden-layer ReLU network, Adam optimizer, fixed-seed mini-batches."""
    def __init__(self,width=16,epochs=20,seed=42):
        self.width=width;self.epochs=epochs;self.seed=seed
    def fit(self,X,y):
        X=np.asarray(X,dtype=float);y=np.asarray(y,dtype=float)
        self.mean=X.mean(0);self.scale=X.std(0);self.scale[self.scale==0]=1
        X=np.clip((X-self.mean)/self.scale,-10,10)
        self.ym=y.mean();self.ys=max(y.std(),1e-8);y=(y-self.ym)/self.ys
        rng=np.random.default_rng(self.seed)
        self.params=[rng.normal(0,np.sqrt(2/X.shape[1]),(X.shape[1],self.width)),np.zeros(self.width),rng.normal(0,.1,self.width),np.zeros(1)]
        moments=[np.zeros_like(p) for p in self.params];variances=[np.zeros_like(p) for p in self.params];step=0
        for epoch in range(self.epochs):
            order=rng.permutation(len(X))
            for begin in range(0,len(X),1024):
                idx=order[begin:begin+1024];x=X[idx];t=y[idx]
                w,b,v,c=self.params;h=np.maximum(x@w+b,0);err=h@v+c[0]-t
                # Squared-error objective with weight decay; selection uses validation MAE.
                delta=2*err/len(idx);dh=(delta[:,None]*v)*(h>0)
                gradients=[x.T@dh+1e-4*w,dh.sum(0),h.T@delta+1e-4*v,np.array([delta.sum()])]
                step+=1
                for k,g in enumerate(gradients):
                    g=np.clip(g,-10,10);moments[k]=.9*moments[k]+.1*g;variances[k]=.999*variances[k]+.001*g*g
                    self.params[k]-=.001*(moments[k]/(1-.9**step))/(np.sqrt(variances[k]/(1-.999**step))+1e-8)
        return self
    def predict(self,X):
        x=np.clip((np.asarray(X)-self.mean)/self.scale,-10,10)
        w,b,v,c=self.params
        return (np.maximum(x@w+b,0)@v+c[0])*self.ys+self.ym

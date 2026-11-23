"""NumPy binary classifiers: logistic regression and sigmoid-output ReLU MLP."""
import numpy as np

def sigmoid(z):return 1/(1+np.exp(-np.clip(z,-35,35)))

class LogisticClassifier:
    def fit(self,X,y):
        X=np.asarray(X,dtype=float);y=np.asarray(y,dtype=float)
        self.mean=X.mean(0);self.scale=X.std(0);self.scale[self.scale==0]=1
        z=np.column_stack([np.ones(len(X)),np.clip((X-self.mean)/self.scale,-10,10)])
        self.coef=np.zeros(z.shape[1]);penalty=np.eye(z.shape[1])*10;penalty[0,0]=0
        for _ in range(30):
            p=sigmoid(z@self.coef)
            gradient=z.T@(p-y)+penalty@self.coef
            hessian=z.T@(z*(p*(1-p))[:,None])+penalty+np.eye(z.shape[1])*1e-8
            step=np.linalg.solve(hessian,gradient);self.coef-=step
            if np.max(np.abs(step))<1e-7:break
        return self
    def predict(self,X):
        z=np.column_stack([np.ones(len(X)),np.clip((np.asarray(X)-self.mean)/self.scale,-10,10)])
        return sigmoid(z@self.coef)

class NeuralClassifier:
    def __init__(self,width=16,epochs=20,seed=42):self.width=width;self.epochs=epochs;self.seed=seed
    def fit(self,X,y):
        X=np.asarray(X,dtype=float);y=np.asarray(y,dtype=float)
        self.mean=X.mean(0);self.scale=X.std(0);self.scale[self.scale==0]=1
        X=np.clip((X-self.mean)/self.scale,-10,10);rng=np.random.default_rng(self.seed)
        self.params=[rng.normal(0,np.sqrt(2/X.shape[1]),(X.shape[1],self.width)),np.zeros(self.width),rng.normal(0,.1,self.width),np.array([np.log((y.mean()+1e-6)/(1-y.mean()+1e-6))])]
        moments=[np.zeros_like(p) for p in self.params];variances=[np.zeros_like(p) for p in self.params];step=0
        for _ in range(self.epochs):
            order=rng.permutation(len(X))
            for begin in range(0,len(X),1024):
                idx=order[begin:begin+1024];x=X[idx];t=y[idx]
                w,b,v,c=self.params;h=np.maximum(x@w+b,0)
                delta=(sigmoid(h@v+c[0])-t)/len(idx);dh=(delta[:,None]*v)*(h>0)
                gradients=[x.T@dh+1e-4*w,dh.sum(0),h.T@delta+1e-4*v,np.array([delta.sum()])]
                step+=1
                for k,g in enumerate(gradients):
                    moments[k]=.9*moments[k]+.1*g;variances[k]=.999*variances[k]+.001*g*g
                    self.params[k]-=.001*(moments[k]/(1-.9**step))/(np.sqrt(variances[k]/(1-.999**step))+1e-8)
        return self
    def predict(self,X):
        x=np.clip((np.asarray(X)-self.mean)/self.scale,-10,10);w,b,v,c=self.params
        return sigmoid(np.maximum(x@w+b,0)@v+c[0])

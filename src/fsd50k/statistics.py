"""Stable train-only sufficient statistics and grouped-threshold average precision."""
import numpy as np
from ._io import npz, load
from scipy.linalg import blas,eigh
LAMBDAS=np.array([1e-4,1e-3,1e-2,1e-1,1.],dtype=np.float64)
STD_EPS=1e-12

class Statistics:
    def __init__(self,d,c=200):
        self.n=0;self.mx=np.zeros(d);self.my=np.zeros(c)
        self.xx=np.zeros((d,d),dtype=np.float64,order='F');self.xy=np.zeros((d,c))
    def add(self,x,y):
        x=np.asarray(x,np.float64);y=np.asarray(y,np.float64);assert len(x)==len(y)>0
        n=len(x);mx=x.mean(0);my=y.mean(0);a=x-mx;b=y-my;dx=mx-self.mx;dy=my-self.my;total=self.n+n
        self.xx=blas.dsyrk(1.,np.asfortranarray(a.T),beta=1.,c=self.xx,lower=1,overwrite_c=1)
        self.xx=blas.dsyr(self.n*n/total,dx,a=self.xx,lower=1,overwrite_a=1)
        self.xy+=a.T@b+np.outer(dx,dy)*(self.n*n/total)
        self.mx+=dx*n/total;self.my+=dy*n/total;self.n=total
    def remove(self,x,y):
        assert self.n>1
        dx=np.asarray(x,np.float64)-self.mx;dy=np.asarray(y,np.float64)-self.my
        self.xx=blas.dsyr(-self.n/(self.n-1),dx,a=self.xx,lower=1,overwrite_a=1)
        self.xy-=np.outer(dx,dy)*self.n/(self.n-1)
        self.mx-=dx/(self.n-1);self.my-=dy/(self.n-1);self.n-=1
    def save(self,path):
        npz(path,n=self.n,mx=self.mx,my=self.my,xx=self.xx,xy=self.xy)
    @classmethod
    def load(cls,path):
        a=load(path);s=cls(len(a['mx']),len(a['my']));s.n=int(a['n']);s.mx=a['mx'];s.my=a['my'];s.xx=np.asfortranarray(a['xx']);s.xy=a['xy'];return s
    def factors(self):
        # LAPACK reads only lower triangle. Normalize in place without a D squared outer-product allocation.
        std=np.sqrt(np.maximum(np.diag(self.xx)/self.n,0));std[std<STD_EPS]=1.
        for j in range(len(std)):self.xx[j:,j]/=std[j:]*std[j]
        self.xy/=std[:,None]
        w,v=eigh(self.xx,lower=True,overwrite_a=True,check_finite=False,driver='evr')
        tol=1e-9*max(1.,abs(w).max());assert w.min()>=-tol,'GRAM_NOT_PSD'
        projected=v.T@self.xy
        return dict(n=self.n,mean=self.mx,std=std,target_mean=self.my,eigenvalues=np.maximum(w,0),eigenvectors=v,projected=projected)

def coefficient(f,lam):
    return f['eigenvectors']@(f['projected']/(f['eigenvalues'][:,None]+f['n']*lam))
def predict(x,f,w):return ((np.asarray(x,np.float64)-f['mean'])/f['std'])@w+f['target_mean']

class APOrder:
    def __init__(self,labels,scores):
        self.labels=np.asarray(labels,np.float64);scores=np.asarray(scores)
        assert scores.shape==self.labels.shape and np.isfinite(scores).all() and np.isin(self.labels,[0,1]).all()
        self.order=np.argsort(-scores,kind='stable');s=scores[self.order]
        self.ends=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1];self.y=self.labels[self.order]
    def value(self,weights=None):
        weights=np.ones(len(self.y)) if weights is None else np.asarray(weights,np.float64)[self.order]
        pos=np.sum(self.y*weights);neg=np.sum((1-self.y)*weights)
        if pos<=0 or neg<=0:return np.nan
        tp=np.cumsum(self.y*weights)[self.ends];total=np.cumsum(weights)[self.ends]
        precision=np.divide(tp,total,out=np.zeros_like(tp),where=total>0)
        return float(np.dot(np.diff(np.r_[0.,tp]),precision)/pos)

class MultilabelAP:
    def __init__(self,y,scores):
        assert y.ndim==scores.ndim==2 and y.shape==scores.shape
        self.y=y;self.orders=[APOrder(y[:,j],scores[:,j]) for j in range(y.shape[1])]
    def values(self,weights=None):return np.array([o.value(weights) for o in self.orders])
    def report(self,scores):
        ap=self.values();supports=self.y.sum(0).astype(int)
        return dict(per_class_ap=[None if not np.isfinite(v) else float(v) for v in ap],positive_support=supports.tolist(),negative_support=(len(self.y)-supports).tolist(),defined_classes=int(np.isfinite(ap).sum()),macro_200_ap=float(ap.mean()) if np.isfinite(ap).all() else None,defined_class_macro_ap=float(np.nanmean(ap)) if np.isfinite(ap).any() else None,micro_ap=APOrder(self.y.ravel(),scores.ravel()).value())

def group_components(rows):
    # Official eval uploader grouping further merged by exact content duplicates.
    parent=list(range(len(rows)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    for i,r in enumerate(rows):
        for key in [('uploader',str(r['uploader'])),('hash',r['content_hash_group'])]:
            if key in seen:parent[root(i)]=root(seen[key])
            else:seen[key]=i
    roots=[root(i) for i in range(len(rows))];mapping={v:j for j,v in enumerate(sorted(set(roots)))}
    return np.array([mapping[v] for v in roots],np.int32)

def bootstrap(y,score_map,groups,draws=2000,seed=17170911):
    orders={k:MultilabelAP(y,v) for k,v in score_map.items()};rng=np.random.default_rng(seed);n=int(groups.max())+1
    values={k:np.full(draws,np.nan) for k in score_map};missing=np.zeros(draws,dtype=np.int32)
    for b in range(draws):
        counts=np.bincount(rng.integers(n,size=n),minlength=n);weights=counts[groups]
        pos=weights@y;neg=weights.sum()-pos;missing[b]=int(((pos<=0)|(neg<=0)).sum())
        for k,o in orders.items():
            ap=o.values(weights)
            if np.isfinite(ap).all():values[k][b]=ap.mean()
    return values,missing

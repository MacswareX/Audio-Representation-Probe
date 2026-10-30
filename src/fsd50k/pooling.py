"""Balanced exact sample coverage and stable physical-time weighted pooling."""
import numpy as np
def boundaries(frames,sr=44100):
    n=(frames+5*sr-1)//(5*sr);b=np.array([j*frames//n for j in range(n+1)],dtype=np.int64)
    assert b[0]==0 and b[-1]==frames and (np.diff(b)>0).all() and np.diff(b).max()<=5*sr
    if n>1:assert np.diff(b).min()>2.5*sr-1
    return b
def mapped_boundaries(original,sr,target_sr,target_frames):
    # Floor original physical time onto target sample clock; final endpoint matches native resampler ceil.
    b=original*target_sr//sr;b[-1]=target_frames;assert (np.diff(b)>0).all();return b
def voronoi(anchors,duration):
    anchors=np.asarray(anchors,np.float64);assert len(anchors)>0 and ((anchors>=0)&(anchors<=duration+1e-9)).all() and (np.diff(anchors)>0).all()
    anchors=np.minimum(anchors,duration)
    edges=np.r_[0.,(anchors[:-1]+anchors[1:])/2,duration];assert (np.diff(edges)>0).all()
    return np.stack([edges[:-1],edges[1:]],1)
class Moments:
    def __init__(self,d):self.w=0.;self.mean=np.zeros(d);self.m2=np.zeros(d);self.cells=0
    def add(self,x,w):
        x=np.asarray(x,np.float64);w=np.asarray(w,np.float64);keep=w>0;x=x[:,keep];w=w[keep]
        if not len(w):return
        mass=w.sum();mean=(x*w).sum(1)/mass;m2=((x-mean[:,None])**2*w).sum(1);delta=mean-self.mean;total=self.w+mass
        self.m2+=m2+delta**2*(self.w*mass/total);self.mean+=delta*(mass/total);self.w=total;self.cells+=len(w)
    def vector(self):
        assert self.w>0;return np.r_[self.mean,np.sqrt(np.maximum(self.m2/self.w,0))].astype(np.float32)
class ClipPool:
    def __init__(self,d,duration):self.duration=duration;self.g=Moments(d);self.t=[Moments(d) for _ in range(4)]
    def add(self,x,anchors,start,duration):
        cells=voronoi(anchors,duration)+start;self.g.add(x,cells[:,1]-cells[:,0])
        for j,m in enumerate(self.t):
            lo=self.duration*j/4;hi=self.duration*(j+1)/4;w=np.maximum(0,np.minimum(cells[:,1],hi)-np.maximum(cells[:,0],lo));m.add(x,w)
    def result(self):
        assert abs(self.g.w-self.duration)<1e-7 and all(abs(m.w-self.duration/4)<1e-7 for m in self.t)
        return self.g.vector(),np.concatenate([m.vector() for m in self.t]),dict(duration=self.duration,global_weight=self.g.w,bin_weights=[m.w for m in self.t],bin_cell_counts=[m.cells for m in self.t])
def fixtures():
    cases=[1,13230,220499,220500,220501,441000,24266000]
    for n in cases:
        b=boundaries(n);assert np.diff(b).sum()==n
        for sr in (16000,24000,32000):
            mapped=mapped_boundaries(b,44100,sr,(n*sr+44099)//44100)
            assert np.diff(mapped).sum()==mapped[-1]
    p=ClipPool(2,.3);p.add(np.array([[2.],[3.]]),np.array([.075]),0,.3);g,t,rec=p.result();assert np.array_equal(g,[2,3,0,0]) and np.array_equal(t,np.tile(g,4))
    p=ClipPool(2,6.);x=np.array([[1.,2.,3.],[3.,4.,5.]])
    p.add(x,[.5,1.5,2.5],0,3);p.add(x,[.5,1.5,2.5],3,3);g,t,rec=p.result();assert np.allclose(g[:2],[2,4])
    return dict(exact_sample_partition=True,resampled_clock_partition=True,single_cell_all_bins_valid=True,weighted_stable_merge=True)

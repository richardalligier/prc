import os
from argparse import Namespace
import argparse
from subprocess import call
import random
import time

import numpy as np




class ScoreDataBase:
    def __init__(self,fname,match,score):
        self.fname = fname
        self.match = match
        self.score = score
        self.results = []
    def update(self):
        try:
            with open(self.fname) as f:
                self.results = eval(f"[{f.read()}]")
            # print(self.results)
        except Exception as e:
            # raise e
            print(f"could not read {self.fname}")
    def search(self,x):
        for d in self.results:
            if self.match(d,x):
                return self.score(d)

class Grid:
    def __init__(self, grid):
        self.grid = {k:list(v) for k,v in grid.items()}
    def to_coord(self,which):
        return {k:self.grid[k].index(which[k]) for k in self.grid}
    def from_coord(self,x):
        return {k:self.grid[k][x[k]] for k in self.grid}
    def to_neigh_coord(self,x,k):
        n = len(self.grid[k])
        assert(k in x)
        i = x[k]
        l = [j for j in [i-1,i+1] if 0<=j<n]
        res = []
        for j in l:
            res.append(x.copy())
            res[-1][k]=j
        return res
    def random_neigh_coord(self,x,nk=1):
        candidatek = [k for k in self.grid if len(self.grid[k])>1]
        selectedk = random.sample(candidatek,nk)
        # print(selectedk)
        return list(newx for k in selectedk for newx in self.to_neigh_coord(x,k))
    def random_neigh(self,params,nk=1):
        coords = self.to_coord(params)
        neighs =[self.from_coord(x) for x in  self.random_neigh_coord(coords)]
        for nc in neighs:
            for k in params:
                if k not in nc:
                    nc[k]=params[k]
        return neighs



class GridRandom:
    def __init__(self, grid):
        self.grid = grid

    def random_neigh(self,params,nk=1):
        neighs = [{k:v() for k,v in self.grid.items()} for i in range(2)]
        for i,nc in enumerate(neighs):
            for k in params:
                if k not in nc:
                    nc[k]=params[k]
            nc["random_state"]=i
        return neighs



def search(scoredb,grid,params,f):
    while True:
        lparams = grid.random_neigh(params,nk=2)
        lres = []
        for params in lparams:
            cscore = scoredb.search(params)
            if cscore is None:
                f(params)
                time.sleep(0.1)
                scoredb.update()
                cscore = scoredb.search(params)
                assert (cscore is not None)
            lres.append(cscore)
        print(lres)
        params = min(zip(lres,enumerate(lparams)))[1][1]
        print(params)



def main():
    parser = argparse.ArgumentParser(
                    description='Hyperparameter optimizer through random search',
    )
    args = parser.parse_args()
    dicoparamRandom = {
        "learning_rate":lambda : np.random.uniform(0.03,0.07),
        "num_leaves":lambda : np.random.randint(10,21),
        "max_bin":lambda : 2**np.random.randint(6,10)-1,
        "cat_l2": lambda : 50*np.random.randint(0,10),
        "cat_smooth":lambda : 50*np.random.randint(0,10),
        "reg_lambda":lambda : 100*np.random.randint(0,10),
        "reg_alpha":lambda : 100*np.random.randint(0,10),
        "subsample":lambda : np.random.uniform(0.6,1),
        "colsample_bytree":lambda : np.random.uniform(0.6,1),
        "min_child_samples":lambda : 20*np.random.randint(0,20),
    }
    model_params = {
        "n_estimators":50000,
        "random_state":0,
        "device":"cpu",
        "verbose":1,
        "max_depth":-1,
        "learning_rate":0.05,
        "num_leaves":17,
        "max_bin":127,
        "cat_l2":50,
        "cat_smooth":50,
        "reg_lambda":700,
        "reg_alpha":700,
        'subsample_for_bin': 400000,
        "subsample":1,
        "colsample_bytree":1,
        "min_child_samples":20,
        'min_child_weight': 0.,
        'importance_type':'gain',
        'min_split_gain':0.,
    }
    grid = GridRandom(dicoparamRandom)
    fname = "logsearch"
    def f(x):
        x = x.copy()
        if x["subsample"]<1:
            x["subsample_freq"]=100
        cmd = f"python3 regression.py"
        for k,v in x.items():
            cmd += f" -{k} {v}"
        print(cmd)
        call(cmd,shell=True)
    def match(d,x):
        for k,v in x.items():
            if d["model_params"][k]!=v:
                return False
        return True
    def score(d):
        return d["test_error"]
    scoredb = ScoreDataBase(fname,match,score)
    search(scoredb,grid,model_params,f)


if __name__ == '__main__':
    main()

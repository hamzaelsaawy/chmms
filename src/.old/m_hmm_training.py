import numpy as np
#import pandas as pd
import sklearn as skl
from hmmlearn import hmm
from sklearn.externals import joblib

#
# load data
#

DATA_DIR = '../../data/trajdata_i101_trajectories-0750am-0805am/'
traj_lens = np.loadtxt(DATA_DIR + 'traj_lengths.csv', dtype=np.int)
D = np.loadtxt(DATA_DIR + 'X_disc.csv', delimiter=',', dtype=np.int)

n_components = 5
n_samples = D.shape[0]
n_trajs = len(traj_lens)

#
# states are only seen observations
#
obs = set()
for i in range(n_samples):
    obs.add(tuple(D[i]))

obs_lut = dict()
for (i, o) in enumerate(obs):
    obs_lut[o] = i

n_features = len(obs_lut)
X = np.apply_along_axis(lambda o: obs_lut[tuple(o)], axis=1, arr=D)
X = X[:, np.newaxis]

for i in range(10):
    print(i, end=' ')
    model = hmm.MultinomialHMM(n_components=n_components, n_iter=100,
            transmat_prior=np.random.dirichlet(np.repeat(1, n_components), n_components))
    model.n_features = n_features
    model.fit(X, lengths=traj_lens)

    loglike = model.monitor_.history[1]
    joblib.dump(model, DATA_DIR + 'm' + str(abs(int(loglike))) + '.pkl')
    print(loglike)


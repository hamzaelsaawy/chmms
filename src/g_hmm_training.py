import numpy as np
#import pandas as pd
import sklearn as skl
from hmmlearn import hmm
from sklearn.externals import joblib
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

#
# load data
#
DATA_DIR = '../../data/trajdata_i101_trajectories-0750am-0805am/'
traj_lens = np.loadtxt(DATA_DIR + 'traj_lengths.csv', dtype=np.int)
X = np.loadtxt(DATA_DIR + 'X.csv', delimiter=',')
X = X[:, :2] # drop off lane data

n_components = 8
n_trajs = len(traj_lens)
n_samples = X.shape[0]

for i in range(20):
    print(i, end=' ')
    model = hmm.GaussianHMM(n_components=n_components, verbose=False,
           n_iter=75, covariance_type='full')
    #model.transmat_ = np.random.dirichlet(np.repeat(1, n_components), n_components)
    model.fit(X, lengths=traj_lens)

    loglike = model.monitor_.history[1]
    joblib.dump(model, DATA_DIR + 'g' + str(abs(int(loglike))) + '.pkl')
    print(loglike)


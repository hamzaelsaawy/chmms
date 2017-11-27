import ProfileView

include("src/CHMM.jl")
using CHMM

#
# Data Gen
#
KK = 4
DD = 2

model = rand_chmm(KK, DD)
(X, Z, trajptr_full, pairs_full) = rand_trajs(model, T_range=50:50, N_pairs=500)
# total # trajs*2 = total # pairs
num_trajs_full = length(trajptr_full) - 1
num_pairs_full = size(pairs_full, 2)

# cut all the data in half, otherwise P and S refer to the same data
# by half, num_trajs = num_pairs, so P contains twice as much data...
# add 1 to traj ptr b/c it needs the end of the last traj
trajptr = trajptr_full[1:round(Int, num_trajs_full/3)]
pairs = pairs_full[:, round(Int, num_pairs_full/3)+1:end]
num_trajs = length(trajptr) - 1
num_pairs = size(pairs, 2)

num_obs = size(X, 2)

model_ll(model, X, trajptr, pairs);

@time model_ll(model, X, trajptr, pairs);

Profile.init(n=10_000_000, delay=0.005)
Profile.clear()
#Profile.clear_malloc_data()
@profile model_ll(model, X, trajptr, pairs);

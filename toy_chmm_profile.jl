include("src/CHMM.jl")
using CHMM

#
# Data Gen
#
KK = 4
DD = 2

model = rand_chmm(KK, DD)
(X, Z, trajptr_full, pairs_full) = rand_trajs(model, T_range=200:500, N_pairs=5_000)
num_trajs_full = length(trajptr_full) - 1
num_pairs_full = size(pairs_full, 2)

# cut all the data in half, otherwise P and S refer to the same data
trajptr = trajptr_full[1:round(Int, num_trajs_full/2)]
pairs = pairs_full[:, round(Int, num_pairs_full/2)+1:end]
num_trajs = length(trajptr) - 1
num_pairs = size(pairs, 2)

num_obs = size(X, 2)
println("$(num_obs) observations total")

#
# EM
#
#model_ll_data[i] = model_ll(model, X, trajptr, pairs)

#curr = chmm_from_data(X, KK)
#suff = ChmmSuffStats(curr)
#(curr, _) = chmm_em!(curr, suff, X, trajptr, pairs; N_iters=N_iters, print_every=10)


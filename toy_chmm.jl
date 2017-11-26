using DataFrames
import CSV

include("src/CHMM.jl")
using CHMM

#
# Data Gen
#
KK = 4
DD = 2

model = rand_chmm(KK, DD)
(X, Z, trajptr_full, pairs_full) = rand_trajs(model,
    T_range=200:500, N_pairs=5_000)
num_trajs_full = length(trajptr_full) - 1
num_pairs_full = size(pairs_full, 2)

# cut all the data in half, otherwise P and S refer to the same data
trajptr = trajptr_full[1:round(Int, num_trajs_full/2)]
pairs = pairs_full[:, round(Int, num_pairs_full/2)+1:end]
num_trajs = length(trajptr) - 1
num_pairs = size(pairs, 2)

num_obs = size(X, 2)

#
# EM trials
#
N_trials = 5
N_sizes = 6
N_iters = 100

perm = randperm(num_pairs)
P_sizes = ceil.(Int, num_pairs * linspace(0, 1, N_sizes))
model_ll_data = zeros(Float64, N_sizes)
curr_ll_data = zeros(Float64, N_trials, N_sizes)
;

for (i, P_size) in enumerate(P_sizes)
    println("\n $i: $P_size")
    pairs_trimmed = pairs[:, perm[1:P_size]]

    model_ll_data[i] = model_ll(model, X, trajptr, pairs_trimmed)

    for n in 1:N_trials
        println(" iteration: $n")
        curr = chmm_from_data(X, KK)
        suff = ChmmSuffStats(curr)
        (curr, _) = chmm_em!(curr, suff, X, trajptr, pairs_trimmed;
                N_iters=N_iters, print_every=10)

        curr_ll_data[n, i] = model_ll(curr, X, trajptr, pairs_trimmed)
    end
end

ll_data = DataFrame()
ll_data[:P_size] = P_sizes
ll_data[:model] = model_ll_data
[ll_data[Symbol("train_$i")] = curr_ll_data[i, :] for i in 1:N_trials]

CSV.write("data/ll_$(N_trials)_trials_$(N_sizes)_$(num_obs).csv", ll_data)


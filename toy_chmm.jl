using DataFrames
import CSV

include("src/CHMM.jl")
using CHMM

#
# Data Gen
#
KK = 4
DD = 2

traj_range = 200:500
N_pairs = 500

model = rand_chmm(KK, DD)
(X, Z, trajptr_full, pairs_full) = rand_trajs(model, T_range=traj_range, N_pairs=N_pairs)
# total # trajs*2 = total # pairs
num_trajs_full = length(trajptr_full) - 1
num_pairs_full = size(pairs_full, 2)
num_obs = size(X, 2)

println(num_obs, num_pairs_full, num_trajs_full)

#
# EM trials
#

ratio_range = (0.1,10) # ranges of P/(2 S)
N_sizes = 5 # different sizes of P/(2 S)
N_trials = 6 # number of trials per size
N_iters = 150 # max number of iterations per model

ranges = linspace(ratio_range..., N_sizes)
P_sizes = ceil.(Int, ranges./(1.+ranges) * num_pairs_full)

model_ll_data = zeros(Float64, N_sizes)
curr_ll_data = zeros(Float64, N_trials, N_sizes)

for (i, P_size) in enumerate(P_sizes)
    println("\n $i/$(N_sizes): $P_size")

    # (+1) to get the end of the last one
    trajptr = trajptr_full[1:2*(num_pairs_full - P_size) + 1]
    pairs = pairs_full[:, end-P_size+1:end]

    model_ll_data[i] = model_ll(model, X, trajptr, pairs)

    for n in 1:N_trials
        println(" iteration: $n/$(N_trials)")

        curr = chmm_from_data(X, KK)
        suff = ChmmSuffStats(curr)
        (curr, _) = chmm_em!(curr, suff, X, trajptr, pairs;
                N_iters=N_iters, print_every=10)

        curr_ll_data[n, i] = model_ll(curr, X, trajptr, pairs)
    end
end

ll_data = DataFrame()
ll_data[:P_size] = P_sizes
ll_data[:model] = model_ll_data
[ll_data[Symbol("train_$i")] = curr_ll_data[i, :] for i in 1:N_trials]

CSV.write("data/ll_$(num_pairs_full)_pairs_$(N_iters)_iters.csv", ll_data)

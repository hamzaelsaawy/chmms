using DataFrames
import CSV

include("src/CHMM.jl")

const data_path = "data/toy_data"

dfX = CSV.read(joinpath(data_path, "X.csv"), header=true, nullable=false)
Xfull = Matrix(dfX)
X = Matrix(dfX[[:velocity, :mindistance, :acceleration]])'

trajptr = readcsv(joinpath(data_path, "traj_ptr.csv"), Int) |> vec
pairsfull, _ = readcsv(joinpath(data_path, "pairs.csv"), Int, header=true)
pairsfull = pairsfull'

n_trajs = length(trajptr) - 1
n_pairs = size(pairsfull, 2)
n_obs = size(X, 2)

include("src/CHMM.jl")
X = X .* [1, 1, 100] # scaling issue for acceleration

K = 3
curr = chmm_from_data(X, K)
orig_est = deepcopy(curr)
suff = ChmmSuffStats(curr)

prinln("Starting CHMM training ...")

(curr, loglike_hist) = chmm_em!(curr, suff, X, trajptr, pairsfull;
        N_iters=50, print_every=10)
;


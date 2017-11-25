#
# generation and stuff...
#

export
    Chmm,
    chmm_from_data,
    rand_chmm,
    simulate_model,
    rand_trajs,
    model_ll

struct Chmm
    K::Int # number states
    D::Int
    π0::Vector{Float64} # K elements (not K²)
    P::Array{Float64,3} # the K×K×K (factored) distribution
    μs::Vector{Vector{Float64}}
    Σs::Vector{Matrix{Float64}}
end

"""
    chmm_from_data(X::Matrix{<:Real}, K::Int; maxiter::Int=50, display::Symbol=:none)

initial EM estimates (just does k-means)
X is D×T, where D is data dimension
"""
function chmm_from_data(X::Matrix{<:Real}, K::Int;
        maxiter::Int=50, display::Symbol=:none) # kmeans params
    D = size(X, 1)

    R = kmeans(X, K, maxiter=maxiter, display=display)
    ms = [R.centers[:, i] for i in 1:K]
    Ss = [eye(D) for _ in 1:K]

    p0 = normalize(rand(K), 1)

    P = rand(K, K, K)
    P ./= sum(P, 1)

    return Chmm(K, D, p0, P, ms, Ss)
end

"""
    rand_chmm(K::Int=5, D::Int=3, μ_scale::Real=10, Σ_scale::Real=1)
"""
function rand_chmm(K::Int=5, D::Int=3, μ_scale::Real=10, Σ_scale::Real=1)
    π0 = rand(K)
    π0 ./= sum(π0)

    # p(zᵗ⁺¹ | z₁ᵗ, z₂ᵗ) = P[zᵗ⁺¹, z₁ᵗ, z₂ᵗ]
    # Z₁ ≡ Z₂
    P = rand(K, K, K)
    P ./= sum(P, 1)

    μs = [randn(D) * μ_scale for _ in 1:K]
    Σs = [eye(D) * rand() * Σ_scale for _ in 1:K]

    return Chmm(K, D, π0, P, μs, Σs)
end

function simulate_model(model::Chmm, T::Int=500)
    D = model.D
    K = model.K
    KK = K^2

    Z = empty(Int, 2, T)
    X1 = empty(D, T)
    X2 = similar(X1)

    sqrtm_Σs = map(sqrtm, model.Σs)
    z = wsample(1:K, model.π0, 2)

    for t in 1:T
        z1 = wsample(1:K, model.P[:, z...])
        z2 = wsample(1:K, model.P[:, reverse(z)...])

        Z[:, t] = [z1, z2]
        X1[:, t] = sqrtm_Σs[z1] * randn(D) + model.μs[z1]
        X2[:, t] = sqrtm_Σs[z2] * randn(D) + model.μs[z2]
    end

    return (Z, X1, X2)
end

function rand_trajs(model::Chmm; T_range::Range{Int64}=750:1_000, N_pairs::Int=5)
    N_trajs = N_pairs * 2
    trajs = [simulate_model(model, rand(T_range)) for _ in 1:N_pairs]

    traj_lens = map(t -> size(t[1], 2), trajs)
    num_obs = sum(traj_lens) * 2

    X = empty(model.D, num_obs)
    Z = empty(Int, num_obs)
    trajptr = empty(Int, N_trajs + 1)
    trajptr[1] = 1
    traj_pairs = empty(Int, 4, N_pairs)

    idx = 1
    # concatenate all the data into a single stream
    for n in 1:N_pairs
        (Zt, X1, X2) = trajs[n]
        T = size(Zt, 2)

        # the trajectory number
        id1 = sub2ind((2, N_pairs), 1, n)
        id2 = id1 + 1

        # start and stop indices in global data
        start_1 = idx
        end_1 = start_1 + T - 1
        start_2 = end_1 + 1
        end_2 = start_2 + T - 1

        Z[start_1:end_1] = Zt[1, :]
        Z[start_2:end_2] = Zt[2, :]
        X[:, start_1:end_1] = X1
        X[:, start_2:end_2] = X2

        # start of each traj
        trajptr[id2] = start_2
        # edge case for last entry
        trajptr[id2 + 1] = end_2 + 1
        # record the pairs
        traj_pairs[:, n] = [start_1, end_1, start_2, end_2]

        idx = end_2 + 1
    end

    return (X, Z, trajptr, traj_pairs)
end


"""
    model_ll(chmm::Chmm, X::Matrix{Float64}, trajptr::Vector{Int}, pairs::Matrix{Int})

Model log-likelihood given data X.
"""
function model_ll(chmm::Chmm,
        X::Matrix{Float64},
        trajptr::Vector{Int},
        pairs::Matrix{Int})
    K = chmm.K
    KK = K^2
    num_trajs = length(trajptr) - 1
    num_pairs = size(pairs, 2)

    log_p0 = log.(vec(outer(chmm.π0)))
    log_P = log.(make_flat(chmm.P))

    T_max = ( length(trajptr) == 1 ) ? maximum(diff(pairs[1:2, :])) + 1: maximum(diff(trajptr))

    log_b = empty(KK, T_max)
    log_α = similar(log_b)
    log_β = similar(log_b)
    γ = similar(log_b)

    loglike = 0.0
    normals = gen_normals(chmm)

    #
    # Single Trajectories
    #
    for id in 1:num_trajs # all trajectories
        Xt = get_trajectory_from_ptr(X, trajptr, id)
        T = size(Xt, 2)

        data_likelihood!(SingleTrajectory, normals, Xt, log_b)
        loglike += forward_backward!(chmm, T,
                log_p0, log_P, log_b, log_α, log_β, γ)
    end

    #
    # Pairwise Trajectories
    #
    for id in 1:num_pairs
        X1, X2 = get_pair_from_ptr(X, pairs, id)
        T = size(X1, 2)
        @assert T == size(X2, 2)

        data_likelihood!(PairwiseTrajectory, normals, X1, X2, log_b)
        loglike += forward_backward!(chmm, T,
                log_p0, log_P, log_b, log_α, log_β, γ)
    end

    return loglike
end

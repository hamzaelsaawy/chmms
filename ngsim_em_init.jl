#
# EM initialization functions and types
#

# code is heavily based on (copied from ?)
# Jeffrey W. Miller (2016). Lecture Notes on Advanced Stochastic Modeling. Duke University, Durham, NC.
# see https://github.com/jwmi/HMM

mutable struct HMM
    K::Int # number states
    M::Int # number gaussian mixtures per state
    L::Int # number of discrete observation states
    # dimension of obs
    N::Int
    NΓ::Int
    NΔ::Int

    A::Matrix{Float64}
    π0::Vector{Float64}
    bΔ::Matrix{Float64}
    c::Matrix{Float64}
    μs::Array{Float64}
    Σs::Array{Float64}
end

function HMM(YΓ::AbstractArray{Float64}, K::Int, M::Int, L::Int, NΓ::Int, NΔ::Int)
    K = 5 # number states
    M = 3 # number gaussian mixtures per state

    # dimension of obs
    N = NΓ + NΔ

    # number of discrete observation states
    # should only be 6 lanes in the data (but just in case)
    L = 7

    #
    # transition parameters
    #

    # transition matrix; logs to avoid numberical underflow
    # dirichlet returns distributions as the columns
    A = rand(Dirichlet(K, 1), K)

    # initial
    π0 = rand(Dirichlet(K, 1))

    #
    # emission parameters
    #

    # bΔ[l, k] = P(yₜ = l | xₜ = k) (rows sum to 1)
    # only for NΔ = 1; should be [rand(...)' for _ in 1:MΔ]
    bΔ = rand(Dirichlet(L, 1), K)

    # mixuter parameters for each MV per state
    # c[m, k] is weight of m-th gaussian in for k-th state
    c = rand(Dirichlet(M, 1), K)

    # μs[:, m, k] is the mean of m-th mixture of the k-th state
    μs = randn(NΓ, M, K) .+ squeeze(mean(YΓ, 2), 2)

    # Σs[:, :, m, k] is the covariance of m-th mixture of the k-th state
    Σs = repeat(cov(YΓ, 2), outer=(1, 1, M, K))

    return HMM(K, M, L, N, NΓ, NΔ, A, π0, bΔ, c, μs, Σs)
end

mutable struct HMM_Data
    T::Int
    E::Int

    S::SparseTrajData
    YΓ::AbstractArray{Float64}
    YΔ::AbstractArray{Int}

    log_A::Matrix{Float64}
    log_π0::Vector{Float64}
    log_bΔ::Matrix{Float64}
    log_c::Matrix{Float64}
    invΣs::Array{Float64}
    logdetΣs::Array{Float64}

    log_α::Array{Float64}
    log_β::Array{Float64}
    log_b::Array{Float64}
    γ::Array{Float64}
    γ_mix::Array{Float64}
    ξ::Array{Float64}
end

function HMM_Data(hmm::HMM, S::SparseTrajData,
        YΓ::AbstractArray{Float64}, YΔ::AbstractArray{Int}, T::Int=0)
    # number of obs
    if T == 0
        T = size(Y, 2)
    end

    # number of trajectories
    E = S.m

    M = hmm.M
    K = hmm.K

    log_A = log.(hmm.A)
    log_π0 = log.(hmm.π0)
    log_bΔ = log.(hmm.bΔ)
    log_c = log.(hmm.c)

    invΣs = similar(hmm.Σs)
    logdetΣs = similar(hmm.Σs, M, K)


    @simd for m in 1:M
        for k in 1:K
            invΣs[:, :, m, k] = inv(hmm.Σs[:, :, m, k])
            logdetΣs[m, k] = logdet(hmm.Σs[:, :, m, k])
        end
    end

    # log P(y₁, ..., yₜ, xₜ | θ)
    log_α = AF64(K, T)
    # log P(yₜ₊₁, ..., y_T | xₜ =i, θ)
    log_β = AF64(K, T)

    # log P(yₜ | xₜ, θ)
    log_b = similar(log_α)

    # P(xₜ = j | Y, θ)
    γ = AF64(K, T)
    # P(Xₜ = j, Zⱼₜ = m| Y, θ) = prob of state j, and m-th mixture at t
    γ_mix = AF64(M, K, T)
    # P(xₜ = j, xₙ₋₁ = i | Y, θ)
    ξ = AF64(K, K, T)

    return HMM_Data(T, E, S, YΓ, YΔ,
        log_A, log_π0, log_bΔ, log_c, invΣs, logdetΣs,
        log_α, log_β, log_b, γ, γ_mix, ξ)
end

function init_em_problem(S::SparseTrajData, fs::AbstractArray{Function},
        K::Int=5, M::Int=3, L::Int=7)
    Y = traj_flat(S, fs...)

    # the continuous data
    YΓ = view(Y, 1:2, :)
    # the discrete data (Int to use as indexing)
    YΔ = round.(Int, Y[3, :])
    NΓ = size(YΓ, 1)
    NΔ = size(YΔ, 1)

    hmm = HMM(YΓ, K, M, L, NΓ, NΔ)
    hmm_data = HMM_Data(hmm, S, YΓ, YΔ, 5_000)

    return hmm, hmm_data
end


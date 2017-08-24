#
# CHMM EM training
#

# wholly unnecessary and pointlessly complicated
abstract type TrajectoryType end
struct PairwiseTrajectory <: TrajectoryType end
struct SingleTrajectory <: TrajectoryType end

"""
Sufficient statstics for a trajectory
"""
struct ChmmSuffStats
    P_flat::Matrix{Float64}
    p0_flat::Vector{Float64}
    counts_K::Vector{Float64}
    counts_KK::Vector{Float64}
    ms::Vector{Vector{Float64}}
    Ss::Vector{Matrix{Float64}}
end

function ChmmSuffStats(model::Chmm)
    K = model.K
    D = model.D
    KK = K^2

    P_flat = zeros(KK, KK)
    p0_flat = zeros(KK)
    counts_K = zeros(K)
    counts_KK = zeros(KK)
    ms = [zeros(D) for _ in 1:K]
    Ss = [zeros(D, D) for _ in 1:K]

    return ChmmSuffStats(P_flat, p0_flat, counts_K, counts_KK, ms, Ss)
end

function zero!(suff::ChmmSuffStats)
    fill!(suff.P_flat, 0)
    fill!(suff.p0_flat, 0)
    fill!(suff.counts_K, 0)
    fill!(suff.counts_KK, 0)

    for k in 1:K
        fill!(suff.ms[k], 0)
        fill!(suff.Ss[k], 0)
    end

    return suff
end

#
# Data Likelihood
#

function data_likelihood!(
        ::Type{PairwiseTrajectory},
        curr::Chmm,
        X1::AbstractMatrix{<:Real},
        X2::AbstractMatrix{<:Real},
        log_p0::Vector{Float64},
        log_P::Matrix{Float64},
        log_b::Matrix{Float64})
    K = curr.K
    T = size(X1, 2)

    l1 = empty(K)
    l2 = empty(K)

    for t in 1:T
        o1 = X1[:, t]
        o2 = X2[:, t]

        for i in 1:K
            l1[i] = logpdf(MvNormal(curr.μs[i], curr.Σs[i]), o1)
            l2[i] = logpdf(MvNormal(curr.μs[i], curr.Σs[i]), o2)
        end

        log_b[:, t] = (l1 .+ l2')[:]
    end
end

function data_likelihood!(
        ::Type{SingleTrajectory},
        curr::Chmm, X::AbstractMatrix{<:Real},
        log_p0::Vector{Float64},
        log_P::Matrix{Float64},
        log_b::Matrix{Float64})
    K = curr.K
    T = size(X, 2)

    for t in 1:T
        o1 = X[:, t]
        for i in 1:K
            reshape(view(log_b, :, t), K, K)[i, :] =
                    logpdf(MvNormal(curr.μs[i], curr.Σs[i]), o1)
        end
    end
end

#
# Forward Backward Algorithm (Baum Welch)
#

function forward_backward!(
        curr::Chmm,
        T::Int,
        log_p0::Vector{Float64},
        log_P::Matrix{Float64},
        log_b::Matrix{Float64},
        log_α::Matrix{Float64},
        log_β::Matrix{Float64},
        γ::Matrix{Float64})
    K = curr.K
    KK = K^2
    temp = empty(KK)

    #
    # forward
    #
    # temp[i] = log_a[i, t-1] + log_A[j, i]
    log_α[:, 1] = log_p0 .+ log_b[:, 1]

    for t in 2:T
        for j in 1:KK
            temp[:] = log_α[:, t-1]
            temp .+= log_P[j, :]
            log_α[j, t] = logsumexp(temp) + log_b[j, t]
        end
    end

    #
    # backward pass
    #
    # temp[j] = log_A[i, j] + log_β[j, t+1] + log_b[j, t+1]
    log_β[:, T] = 0

    for t in (T-1):-1:1
        for i in 1:KK
            temp[:] = log_b[:, t+1]
            temp .+= log_P[:, i]
            temp .+= log_β[:, t+1]
            log_β[i, t] = logsumexp(temp)
        end
    end

    #
    # γ
    #
    for t in 1:T
        γ[:, t] = log_α[:, t]
        γ[:, t] .+= log_β[:, t]
        γ[:, t] .-= logsumexp(γ[:, t])
        γ[:, t] = exp.(γ[:, t])
    end

    return logsumexp(log_α[:, T])
end

#
# Sufficient Statistics
#

# pairwise data is "seen" twice per μ/Σ
observed_states(::Type{PairwiseTrajectory}, x::Vector{<:Real}, K::Int) =
        pair_counts(x, K)

observed_states(::Type{SingleTrajectory}, x::Vector{<:Real}, K::Int) =
        single_counts(x, K)

function _update_suff_stats!(
        traj_type::Type{<:TrajectoryType},
        suff::ChmmSuffStats,
        T::Int,
        log_p0::Vector{Float64},
        log_P::Matrix{Float64},
        γ::Matrix{Float64})
    K = length(suff.counts_K)

    temp_counts = vec(sum(γ[:, 1:T], 2))
    suff.counts_KK .+= temp_counts

    # counts_K is used to normalize μ and Σ
    suff.counts_K .+= observed_states(traj_type, temp_counts, K)

    #
    # P
    #
    temp2 = similar(suff.P_flat)
    for t in 1:(T-1)
        temp2[:] = log_P
        temp2 .+= log_α[:, t]'
        temp2 .+= log_b[:, t+1]
        temp2 .+= log_β[:, t+1]
        temp2 .-= logsumexp(temp2)
        suff.P_flat .+= exp.(temp2)
    end

    #
    # π₀
    #
    suff.p0_flat .+= γ[:, 1]

    return
end

function update_suff_stats!(
        ::Type{PairwiseTrajectory},
        suff::ChmmSuffStats,
        X1::AbstractMatrix{<:Real},
        X2::AbstractMatrix{<:Real},
        log_p0::Vector{Float64},
        log_P::Matrix{Float64},
        γ::Matrix{Float64} )
    K = length(suff.counts_K)
    T = size(X1, 2)

    _update_suff_stats!(PairwiseTrajectory, suff, T, log_p0, log_P, γ)

    #
    # μ & Σ
    #
    for t in 1:T
        p = reshape(γ[:, t], (K, K)) .+ ϵ
        o1 = X1[:, t]
        o2 = X2[:, t]

        for k in 1:K
            pz1 = sum(p[k, :])
            pz2 = sum(p[:, k])

            suff.ms[k] .+= pz1 * o1
            suff.ms[k] .+= pz2 * o2

            suff.Ss[k] .+= pz1 * o1 * o1'
            suff.Ss[k] .+= pz2 * o2 * o2'
        end
    end
end

function update_suff_stats!(
        ::Type{SingleTrajectory},
        suff::ChmmSuffStats,
        X::AbstractMatrix{<:Real},
        log_p0::Vector{Float64},
        log_P::Matrix{Float64},
        γ::Matrix{Float64} )
    K = length(suff.counts_K)
    T = size(X, 2)

    _update_suff_stats!(SingleTrajectory, suff, T, log_p0, log_P, γ)

    #
    # μ & Σ
    #
    for t in 1:T
        p = reshape(γ[:, t], (K, K)) .+ ϵ
        o = X[:, t]

        for k in 1:K
            pz = sum(p[k, :])

            suff.ms[k] .+= pz * o
            suff.Ss[k] .+= pz * o * o'
        end
    end
end

#
# Parameter Updates
#

# ϵ is added to values before normalizing to stop divide by zeros
function update_parameter_estimates!(
        curr::Chmm,
        suff::ChmmSuffStats,
        ϵ::Float64=eps(1.0))
    K = length(suff.counts_K)
    KK = length(suff.counts_KK)

    # swap the indices of y ∈ [1, K²]
    # if T is N×K², then T[:, swapped_indes] is the same as:
    #   R = reshape(T, (N, K, K)
    #   permutedims!(R, R, [1, 3, 2])
    swapped_inds = [rev_ind(i, K) for i in 1:KK]

    suff.P_flat .+= ϵ
    suff.P_flat .+= suff.P_flat[:, swapped_inds]
    suff.P_flat ./= sum(suff.P_flat, 1)

    for k in 1:KK
        i, j = ind2sub((K, K), k)
        curr.P[:, i, j] = estimate_outer(reshape(suff.P_flat[:, k], K, K))
        outer!(reshape(view(log_P, :, k), K, K), curr.P[:, i, j])
    end
    map!(log, log_P, log_P)

    suff.p0_flat .+= ϵ

    curr.π0[:] = estimate_outer(reshape(suff.p0_flat, K, K))
    curr.π0 ./= sum(curr.π0)
    outer!(reshape(log_p0, K, K), curr.π0)
    map!(log, log_p0, log_p0)

    suff.counts_K .+= ϵ
    for k in 1:K
        curr.μs[k][:] = suff.ms[k] ./ (suff.counts_K[k])
        curr.Σs[k][:] = suff.Ss[k] ./ (suff.counts_K[k])
        curr.Σs[k] .-= curr.μs[k] * curr.μs[k]'

        # enforce symmetry
        curr.Σs[k] .+= curr.Σs[k]'
        curr.Σs[k] ./= 2
    end
end


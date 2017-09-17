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

function data_likelihood!(::Type{PairwiseTrajectory},
        normals::Vector{<:MvNormal},
        X1::AbstractMatrix{<:Real},
        X2::AbstractMatrix{<:Real},
        log_b::Matrix{Float64})
    K = length(normals)
    T = size(X1, 2)

    l1 = empty(K)
    l2 = empty(K)

    for t in 1:T
        o1 = X1[:, t]
        o2 = X2[:, t]

        for k in 1:K
            l1[k] = logpdf(normals[k], o1)
            l2[k] = logpdf(normals[k], o2)
        end

        log_b[:, t] = vec(l1 .+ l2')
    end
end

function data_likelihood!(::Type{SingleTrajectory},
        normals::Vector{<:MvNormal},
        X::AbstractMatrix{<:Real},
        log_b::Matrix{Float64})
    K = length(normals)
    T = size(X, 2)

    for t in 1:T
        o = X[:, t]
        r = square_view(log_b, K, :, t)

        for k in 1:K
            r[k, :] = logpdf(normals[k], o)
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
            temp[:] = log_α[:, t - 1]
            temp .+= log_P[j, :]
            log_α[j, t] = logsumexp(temp) + log_b[j, t]
        end
    end

    #
    # backward pass
    #
    # temp[j] = log_A[i, j] + log_β[j, t+1] + log_b[j, t+1]
    log_β[:, T] = 0

    for t in (T - 1):-1:1
        for i in 1:KK
            temp[:] = log_b[:, t + 1]
            temp .+= log_P[:, i]
            temp .+= log_β[:, t + 1]
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

observed_states(::Type{SingleTrajectory}, x::Vector{<:Real}, K::Int) = single_counts(x, K)
# each mean is "seen" twice per time step
observed_states(::Type{PairwiseTrajectory}, x::Vector{<:Real}, K::Int) = pair_counts(x, K)

# shared code between pairwise and single trajs
function _update_suff_stats!(
        traj_type::Type{<:TrajectoryType},
        suff::ChmmSuffStats,
        T::Int,
        log_p0::Vector{Float64},
        log_P::Matrix{Float64},
        log_b::Matrix{Float64},
        log_α::Matrix{Float64},
        log_β::Matrix{Float64},
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
    for t in 1:(T - 1)
        temp2[:] = log_P
        temp2 .+= log_α[:, t]'
        temp2 .+= log_b[:, t + 1]
        temp2 .+= log_β[:, t + 1]
        temp2 .-= logsumexp(temp2)
        suff.P_flat .+= exp.(temp2)
    end

    #
    # π₀
    #
    suff.p0_flat .+= γ[:, 1]

    return
end

function update_suff_stats!(traj_type::Type{SingleTrajectory},
        suff::ChmmSuffStats,
        X::AbstractMatrix{<:Real},
        log_p0::Vector{Float64},
        log_P::Matrix{Float64},
        log_b::Matrix{Float64},
        log_α::Matrix{Float64},
        log_β::Matrix{Float64},
        γ::Matrix{Float64})
    K = length(suff.counts_K)
    T = size(X, 2)

    _update_suff_stats!(traj_type, suff, T,
            log_p0, log_P, log_b, log_α, log_β, γ)
    #
    # μ & Σ
    #
    for t in 1:T
        o = X[:, t]
        p_full = reshape(γ[:, t], K, K)
        p1 = vec(sum(p_full, 2))

        for k in 1:K
            pz = p1[k]

            suff.ms[k] .+= pz * o
            suff.Ss[k] .+= pz * o * o'
        end
    end
end

function update_suff_stats!(traj_type::Type{PairwiseTrajectory},
        suff::ChmmSuffStats,
        X1::AbstractMatrix{<:Real},
        X2::AbstractMatrix{<:Real},
        log_p0::Vector{Float64},
        log_P::Matrix{Float64},
        log_b::Matrix{Float64},
        log_α::Matrix{Float64},
        log_β::Matrix{Float64},
        γ::Matrix{Float64})
    K = length(suff.counts_K)
    T = size(X1, 2)

    _update_suff_stats!(traj_type, suff, T,
            log_p0, log_P, log_b, log_α, log_β, γ)

    #
    # μ & Σ
    #
    for t in 1:T
        o1 = X1[:, t]
        o2 = X2[:, t]

        p_full = reshape(γ[:, t], K, K)
        p1 = vec(sum(p_full, 2))
        p2 = vec(sum(p_full, 1))

        for k in 1:K
            pz1 = p1[k]
            pz2 = p2[k]

            suff.ms[k] .+= pz1 * o1
            suff.ms[k] .+= pz2 * o2

            suff.Ss[k] .+= pz1 * o1 * o1'
            suff.Ss[k] .+= pz2 * o2 * o2'
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
        log_p0::Vector{Float64},
        log_P::Matrix{Float64},
        ϵ::Float64=ϵ)
    K = length(suff.counts_K)
    KK = length(suff.counts_KK)

    suff.P_flat .+= ϵ
    map!(identity, curr.P, suff.P_flat)
    curr.P ./= sum(curr.P, 1)
    map!(log, log_P, curr.P)

    suff.p0_flat .+= ϵ
    map!(identity, curr.π0, suff.p0_flat)
    curr.π0 ./= sum(curr.π0)
    map!(log, log_p0, curr.π0)

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

#
# EM
#
function chmm_em!(
        curr::Chmm,
        suff::ChmmSuffStats,
        X::Matrix{Float64},
        trajptr::Vector{Int},
        pairs::Matrix{Int};
        N_iters::Int=50,
        conv_tol::Real=1e-2,
        verbose::Bool=true,
        print_every::Int=10)
    K = curr.K
    KK = K^2
    num_trajs = length(trajptr) - 1
    num_pairs = size(pairs, 2)

    log_p0 = log.(curr.π0)
    log_P = log.(curr.P)

    T_max = maximum(diff(trajptr))

    log_b = empty(KK, T_max)
    log_α = similar(log_b)
    log_β = similar(log_b)
    γ = similar(log_b)

    loglike_hist = fill(NaN, N_iters)

    for iter in 1:N_iters
        loglike = 0.0
        zero!(suff)

        normals = [MvNormal(curr.μs[i], curr.Σs[i]) for i in 1:K]

        #
        # Single Trajectories
        #
        for id in 1:num_trajs # all trajectories
            Xt = get_trajectory_from_ptr(X, trajptr, id)
            T = size(Xt, 2)

            data_likelihood!(SingleTrajectory, normals, Xt, log_b)
            loglike += forward_backward!(curr, T,
                    log_p0, log_P, log_b, log_α, log_β, γ)
            update_suff_stats!(SingleTrajectory, suff, Xt,
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
            loglike += forward_backward!(curr, T,
                    log_p0, log_P, log_b, log_α, log_β, γ)
            update_suff_stats!(PairwiseTrajectory, suff, X1, X2,
                    log_p0, log_P, log_b, log_α, log_β, γ)
        end

        #
        # combine info from all trajectories
        #
        update_parameter_estimates!(curr, suff, log_p0, log_P)

        loglike_hist[iter] = loglike
        verbose && (iter % print_every == 0) &&
                @printf("iteration %6d:  %.3f\n", iter, loglike)

        if (iter ≥ 2) && (abs(loglike_hist[iter] - loglike_hist[iter - 1]) ≤ conv_tol)
            break
        end
    end

    return curr, loglike_hist
end


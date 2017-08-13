#
# CHMM EM training
#

"""
Sufficient statstics for a trajectory
"""
struct ChmmSuffStats
    P::Matrix{Float64}
    p0::Vector{Float64}
    counts_K::Vector{Float64}
    counts_KK::Vector{Float64}
    ms::Vector{Vector{Float64}}
    Ss::Vector{Matrix{Float64}}
end

function ChmmSuffStats(model::Chmm)
    K = model.K
    D = model.D
    KK = K^2

    P = zeros(KK, KK)
    p0 = zeros(KK)
    counts_K = zeros(K)
    counts_KK = zeros(KK)
    ms = [zeros(D) for _ in 1:K]
    Ss = [zeros(D, D) for _ in 1:K]

    return ChmmSuffStats(P, p0, counts_K, counts_KK, ms, Ss)
end

function zero!(suff_stats::ChmmSuffStats)
    fill!(suff_stats.P, 0)
    fill!(suff_stats.p0, 0)
    fill!(suff_stats.counts_K, 0)
    fill!(suff_stats.counts_KK, 0)

    for k in 1:K
        fill!(suff_stats.ms[k], 0)
        fill!(suff_stats.Ss[k], 0)
    end

    return suff_stats
end

function forward_backward_pair!(curr::Chmm,
        X1::AbstractMatrix{<:Real}, X2::AbstractMatrix{<:Real},
        log_p0::Vector{Float64}, log_P::Matrix{Float64},
        log_b::Matrix{Float64}, log_α::Matrix{Float64},
        log_β::Matrix{Float64}, γ::Matrix{Float64})
    K = curr.K
    KK = K^2

    temp = empty(KK)

    #
    # data likelihood
    #
    for t in 1:T
        o1 = X1[:, t]
        o2 = X2[:, t]
        for q in 1:K^2
            (i, j) = ind2sub((K, K), q)
            log_b[q, t] = logpdf(MvNormal(curr.μs[i], curr.Σs[i]), o1) +
                    logpdf(MvNormal(curr.μs[j], curr.Σs[j]), o2)
        end
    end

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
    end
    map!(exp, γ, γ)

    return logsumexp(log_α[:, T])
end

function update_suff_stats_pair(suff::ChmmSuffStats,
        X1::AbstractMatrix{<:Real}, X2::AbstractMatrix{<:Real},
        log_p0::Vector{Float64}, log_P::Matrix{Float64},
        γ::Matrix{Float64} )
    T = size(X1, 2)
    temp2 = similar(curr.P)

    suff.counts_KK .+= vec(sum(γ[:, 1:T], 2))
    suff.counts_K .+= single_counts(pseudo_counts_KK, K)

    #
    # P
    #
    for t in 1:(T-1)
        temp2[:] = log_P
        temp2 .+= log_α[:, t]'
        temp2 .+= log_b[:, t+1]
        temp2 .+= log_β[:, t+1]
        temp2 .-= logsumexp(temp2)
        suff.P .+= exp.(temp2)
    end

    #
    # π₀
    #
    suff.p0 .+= γ[:, 1]

    #
    # μ & Σ
    #
    for t in 1:T
        p = reshape(γ[:, t], (K, K))
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


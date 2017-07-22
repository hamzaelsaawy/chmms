#
# EM training procedure
#

@inline function log_bΓ(hmm::HMM, hmm_data::HMM_Data,
        m::Int, k::Int, t::Int)
    o::Float64 = 0.0
    lpdf::Float64 = 0.0

    # shamelessly stolen from Distributions.jl/src/multivariate/mvnormal.jl
    lpdf = -(hmm.NΓ * log2π  + hmm_data.logdetΣs[m, k])/2
    o = hmm_data.YΓ[:, t] - hmm.μs[:, m, k]
    lpdf -= dot(o, (hmm_data.invΣs[:, :, m, k] * o)) / 2

    return lpdf
end

@inline function data_likelihood!(hmm::HMM, hmm_data::HMM_Data)
    T = hmm_data.T
    M = hmm.M
    K = hmm.K

    log_b = hmm_data.log_b

    # log_b[t, i] =  p(Yₜ = yₜ | Xₜ = i) = p(YΓₜ = yₜ | Xₜ = i) * p(YΔₜ = yₜ | Xₜ = i)
    lgmm = AF64(M) # log of sum of GMM pdf

    @inbounds for t = 1:T
        for k in 1:K # per state
            lgmm[:] = hmm.c[:, k]
            for m in 1:M # per mixture
                lgmm[m] += logmvnormal(hmm, hmm_data, m, k, t)
            end

            log_b[k, t] = logsumexp(lgmm)
        end

        log_b[:, t] .+= hmm_data.log_bΔ[hmm_data.YΔ[t], :]
    end
end

@inline function forward_backward_pass!(hmm::HMM, hmm_data::HMM_Data)
    T = hmm_data.T
    M = hmm.M
    K = hmm.K

    log_A = hmm_data.log_A
    log_b = hmm_data.log_b
    log_α = hmm_data.log_α
    log_β = hmm_data.log_β

    # forward

    # [ log_a[i, t-1] + log_A[j, i] ]ᵢ
    temp = AF64(K)

    log_α[:, 1] = hmm_data.log_π0 .+ log_b[:, 1]
    @inbounds for t in 2:T
        for j in 1:K
            temp[:] = log_α[:, t-1] + log_A[j, :]
            log_α[j, t] = logsumexp(temp) + log_b[j, t]
        end
    end

    # backward pass
    # temp is [ log_A[i, j] + log_β[j, t+1] + log_b[j, t+1] ]ⱼ

    log_β[:, T] = 0
    @inbounds for t = (T-1):-1:1
        for i in 1:K
            temp[:] = log_A[i, :] + log_b[:, t+1] + log_β[:, t+1]
            log_β[i, t] = logsumexp(temp)
        end
    end
end

function state_probs!(hmm::HMM, hmm_data::HMM_Data)
    T = hmm_data.T
    M = hmm.M
    K = hmm.K

    log_α = hmm_data.log_α
    log_β = hmm_data.log_β
    γ = hmm_data.γ
    γ_mix = hmm_data.γ_mix
    ξ = hmm_data.ξ

    log_A = hmm_data.log_A
    log_b = hmm_data.log_b
    log_c = hmm_data.log_c

    @inbounds for t in 1:T
        γ[:, t] = log_α[:, t] .+ log_β[:, t]
        γ[:, t] -= logsumexp(γ[:, t])
    end
    map!(exp, γ, γ)

    @inbounds for t in 1:T
        # work in logs for a bit
        γ_mix[:, :, t] = log_c

        for k in 1:K # per state
            γ_mix[:, k, t] += log(γ[k, t])
            γ_mix[:, k, t] -= log_b[k, t]

            for m in 1:M # per mixture
                γ_mix[m, k, t] += logmvnormal(hmm, hmm_data, m, k, t)
            end
        end
        γ_mix[:, :, t] -= logsumexp(γ_mix[:, :, t])
    end
    map!(exp, γ_mix, γ_mix)

    @inbounds for t = 1:(T-1)
        ξ[:, :, t] = log_α[:, t]' .+ log_A .+ log_b[:, t+1] .+ log_β[:, t+1]
        ξ[:, : ,t] -= logsumexp(ξ[:, :, t])
    end
    map!(exp, ξ, ξ)

    return
end

#
# model params estimation (considering mult traj's)
#

function update_params!(hmm::HMM, hmm_data::HMM_Data)
    T = hmm_data.T
    M = hmm.M
    K = hmm.K
    E = hmm_data.E

    π0 = hmm.π0
    c = hmm.c
    μs = hmm.μs
    Σs = hmm.Σs

    YΓ = hmm_data.YΓ
    γ = hmm_data.γ
    γ_mix = hmm_data.γ_mix

    fill!(π0, 0)
    for e = 1:E # per trajectory
        # where the e-th trajectory starts in the data
        start_idx = S.colptr[e]

        (start_idx ≥ T) && break

        π0 += γ[:, start_idx]
    end

    normalize!(π0, 1)
    map!(log, hmm_data.log_π0, π0)

    γ_mix_sum = squeeze(sum(γ_mix, 3), 3)
    c[:] = γ_mix_sum ./ sum(γ, 2)'
    c ./= sum(c, 1)
    map!(log, hmm_data.log_c, c)

    fill!(μs, 0)
    @inbounds for t = 1:T
        o = YΓ[:, t]

        for k in 1:K
            for m in 1:M
                μs[:, m, k] += γ_mix[m, k, t] * o
            end
        end
    end

    @inbounds for k in 1:K
        for m in 1:M
            μs[:, m, k] ./= γ_mix_sum[m, k]
        end
    end

    fill!(Σs, 0)
    @inbounds for t = 1:T
        o = YΓ[:, t]

        for k in 1:K
            for m in 1:M
                Σs[:, :, m, k] += γ_mix[m, k, t] * o * o'
            end
        end
    end

    @inbounds for k in 1:K
        for m in 1:M
            Σs[:, :, m, k] ./= γ_mix_sum[m, k]
        end
    end

    @inbounds for m in 1:M
        for k in 1:K
            hmm_data.invΣs[:, :, m, k] = inv(Σs[:, :, m, k])
            hmm_data.logdetΣs[m, k] = logdet(Σs[:, :, m, k])
        end
    end
end


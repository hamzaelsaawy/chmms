#
# EM training procedure
#

@inline function log_bΓ(hmm::HMM, hmm_data::HMM_Data,
        m::Int, k::Int, t::Int)
    o = AF64(hmm.NΓ)
    lpdf::Float64 = 0.0

    # shamelessly stolen from Distributions.jl/src/multivariate/mvnormal.jl
    lpdf = -(hmm.NΓ * log2π  + hmm_data.logdetΣs[m, k])/2
    o = hmm_data.YΓ[:, t] - hmm.μs[:, m, k]
    lpdf -= dot(o, (hmm_data.invΣs[:, :, m, k] * o)) / 2

    return lpdf
end

@inline function data_likelihood!(hmm::HMM, hmm_data::HMM_Data)
    M = hmm.M
    K = hmm.K
    T = hmm_data.T

    # log_b[t, i] =  p(Yₜ = yₜ | Xₜ = i) = p(YΓₜ = yₜ | Xₜ = i) * p(YΔₜ = yₜ | Xₜ = i)
    lgmm = AF64(M) # log of sum of GMM pdf

     for t in 1:T
        for k in 1:K
            lgmm[:] = hmm_data.log_c[:, k]

            for m in 1:M # per mixture
                lgmm[m] += log_bΓ(hmm, hmm_data, m, k, t)
            end

            hmm_data.log_b[k, t] = logsumexp(lgmm)
        end

        hmm_data.log_b[:, t] .+= hmm_data.log_bΔ[hmm_data.YΔ[t], :]
    end
    sanitize_log!(hmm_data.log_b)
end

@inline function forward_backward_pass!(hmm::HMM, hmm_data::HMM_Data)
    M = hmm.M
    K = hmm.K
    T = hmm_data.T

    #
    # forward
    #
    # [ log_a[i, t-1] + log_A[j, i] ]ᵢ
    temp = AF64(K)
    hmm_data.log_α[:, 1] = hmm_data.log_π0 .+ hmm_data.log_b[:, 1]

     for t in 2:T
        for j in 1:K
            temp[:] = hmm_data.log_α[:, t-1]
            temp .+= hmm_data.log_A[j, :]
            hmm_data.log_α[j, t] = logsumexp(temp) + hmm_data.log_b[j, t]
        end
    end

    #
    # backward pass
    #
    # temp is [ log_A[i, j] + log_β[j, t+1] + log_b[j, t+1] ]ⱼ
    hmm_data.log_β[:, T] = 0

     for t in (T-1):-1:1
        for i in 1:K
            temp[:] = hmm_data.log_A[:, i]
            temp .+= hmm_data.log_b[:, t+1]
            temp .+= hmm_data.log_β[:, t+1]
            hmm_data.log_β[i, t] = logsumexp(temp)
        end
    end
end

function state_probs!(hmm::HMM, hmm_data::HMM_Data)
    M = hmm.M
    K = hmm.K
    T = hmm_data.T

    #
    # γ
    #
     for t in 1:T
        hmm_data.γ[:, t] = hmm_data.log_α[:, t]
        hmm_data.γ[:, t] .+= hmm_data.log_β[:, t]
        hmm_data.γ[:, t] .-= logsumexp(hmm_data.γ[:, t])
    end
    map!(exp, hmm_data.γ, hmm_data.γ)
    sanitize!(hmm_data.γ)

    #
    # γ_mix
    #
     for t in 1:T
        # work in logs for a bit
        hmm_data.γ_mix[:, :, t] = hmm_data.log_c
        hmm_data.γ_mix[:, :, t] .+= log.(hmm_data.γ[:, t])'
        hmm_data.γ_mix[:, :, t] .-= hmm_data.log_b[:, t]'
        # subtract (divide out) the discrete prob portion
        # somehow this is sensical and sum(γ_mix, (1,2)) ≈ 1 !!! wut
        hmm_data.γ_mix[:, :, t] .+= hmm_data.log_bΔ[hmm_data.YΔ[t], :]'

        for k in 1:K # per state
            for m in 1:M # per mixture
                hmm_data.γ_mix[m, k, t] += log_bΓ(hmm, hmm_data, m, k, t)
            end
        end
    end
    map!(exp, hmm_data.γ_mix, hmm_data.γ_mix)
    sanitize!(hmm_data.γ_mix)

    #
    # ξ
    #
     for t in 1:(T-1)
        hmm_data.ξ[:, :, t] = hmm_data.log_A
        hmm_data.ξ[:, :, t] .+= hmm_data.log_α[:, t]'
        hmm_data.ξ[:, :, t] .+= hmm_data.log_b[:, t+1]
        hmm_data.ξ[:, :, t] .+= hmm_data.log_β[:, t+1]
        hmm_data.ξ[:, : ,t] .-= logsumexp(hmm_data.ξ[:, :, t])
    end
    map!(exp, hmm_data.ξ, hmm_data.ξ)
    sanitize!(hmm_data.ξ)

    return
end

function update_params!(hmm::HMM, hmm_data::HMM_Data)
    T = hmm_data.T
    M = hmm.M
    K = hmm.K
    NΓ = hmm.NΓ
    E_max = hmm_data.E_max

    γ_mix_sum = squeeze(sum(hmm_data.γ_mix, 3), 3)
    # *should* be equal to sum(γ_mix_sum, 2) ...
    γ_sum = squeeze(sum(hmm_data.γ, 2), 2)

    #
    # π0
    #
    fill!(hmm.π0, 0)

     for e in 1:E_max # per trajectory
        # where the e-th trajectory starts in the data
        start_idx = hmm_data.S.colptr[e]

        hmm.π0 .+= hmm_data.γ[:, start_idx]
    end
    sanitize!(hmm.π0)
    normalize!(hmm.π0, 1)
    map!(log, hmm_data.log_π0, hmm.π0)

    #
    # A
    #
    fill!(hmm.A, 0)
    temp = zeros(K)

     for e in 1:E_max
        e_range = nzrange(hmm_data.S, e)

        for t in e_range
            hmm.A .+= hmm_data.ξ[:, :, t]
            temp .+= hmm_data.γ[:, t]
        end
    end
    hmm.A ./= temp'
    sanitize!(hmm.A)
    map!(log, hmm_data.log_A, hmm.A)

    #
    # bΔ
    #
    fill!(hmm.bΔ, 0.0)

     for t in 1:T
        hmm.bΔ[hmm_data.YΔ[t], :] .+= hmm_data.γ[:, t]
    end
    hmm.bΔ ./= γ_sum'
    sanitize!(hmm.bΔ)
    map!(log, hmm_data.log_bΔ, hmm.bΔ)

    #
    # c
    #
    hmm.c[:] = γ_mix_sum ./ γ_sum'
    sanitize!(hmm.c)
    map!(log, hmm_data.log_c, hmm.c)

    #
    # μs & Σs
    #
    o = AF64(2)
    tt = zeros(NΓ)
    ttt = zeros(NΓ, NΓ)
    fill!(hmm.μs, 0)
    fill!(hmm.Σs, 0)

     for t in 1:T
        o[:] = hmm_data.YΓ[:, t]

        for k in 1:K
            for m in 1:M
                tt[:] = hmm_data.γ_mix[m, k, t]
                tt .*= o
                hmm.μs[:, m, k] += tt

                ttt[:] = hmm_data.γ_mix[m, k, t]
                ttt .*= o
                ttt .*= o'
                hmm.Σs[:, :, m, k] += ttt
            end
        end
    end

     for k in 1:K
        for m in 1:M
            hmm.μs[:, m, k] ./= γ_mix_sum[m, k]
            hmm.Σs[:, :, m, k] ./= γ_mix_sum[m, k]
            hmm.Σs[:, :, m, k] += ϵI
        end
    end

     for k in 1:K
        for m in 1:M
            hmm_data.invΣs[:, :, m, k] = inv(hmm.Σs[:, :, m, k])
            hmm_data.logdetΣs[m, k] = logdet(hmm.Σs[:, :, m, k])
        end
    end
end


function em_iter!(hmm::HMM, hmm_data::HMM_Data)
    data_likelihood!(hmm, hmm_data)
    forward_backward_pass!(hmm, hmm_data)
    state_probs!(hmm, hmm_data)
    update_params!(hmm, hmm_data)
end


function run_em!(hmm::HMM, hmm_data::HMM_Data;
        iters::Int=50, tol::Float64=1e-5, verbose::Bool=false)
    old_A = copy(hmm.A)

    for i in 1:iters
        verbose && print(i, " ")
        em_iter!(hmm, hmm_data)

        # hack in convergence with differnce in A
        δ = sum(abs, old_A .- hmm.A)
        (δ ≤ tol) && break
        old_A = copy(hmm.A)
    end
end


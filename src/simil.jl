#
# Similarity between trajectories
#

function project_A!(Ae::AbstractMatrix{Float64}, hmm::HMM, hmm_data::HMM_Data, e::Int)
    K = hmm.K
    e = min(e, hmm_data.E)
    e_range = hmm_data.S.colptr[e]:min(hmm_data.S.colptr[e+1]-1, hmm_data.T)
    TE = length(e_range)

    (TE ≤ 1) && error("Trajectory $e starts after $(hmm_data.T)")

    # log P(y₁, ..., yₜ, xₜ | θ)
    log_αe = AF64(hmm.K, TE)
    # log P(yₜ₊₁, ..., y_T | xₜ =i, θ)
    log_βe = AF64(K, TE)

    #
    # forward
    #
    # [ log_a[i, t-1] + log_A[j, i] ]ᵢ
    temp = AF64(K)
    log_αe[:, 1] = hmm_data.log_π0 .+ hmm_data.log_b[:, e_range[1]]

    @inbounds for te in 2:TE
        t = e_range[te]
        for j in 1:K
            temp[:] = log_αe[:, te-1]
            temp .+= hmm_data.log_A[j, :]
            log_αe[j, te] = logsumexp(temp) + hmm_data.log_b[j, t]
        end
    end

    #
    # backward pass
    #
    # temp is [ log_A[i, j] + log_β[j, t+1] + log_b[j, t+1] ]ⱼ
    log_βe[:, TE] = 0

    @inbounds for te in (TE-1):-1:1
        t = e_range[te]
        for i in 1:K
            temp[:] = hmm_data.log_A[:, i]
            temp .+= hmm_data.log_b[:, t+1]
            temp .+= log_βe[:, te+1]
            log_βe[i, te] = logsumexp(temp)
        end
    end

    #
    # "project A onto Ae"...
    #
    fill!(Ae, 0)
    temp = zeros(Ae)

    for te in 1:(TE-1)
        t = e_range[te]

        temp[:, :] = hmm_data.log_A
        temp[:, :] .+= log_αe[:, te]'
        temp[:, :] .+= hmm_data.log_b[:, t+1]
        temp[:, :] .+= log_βe[:, te+1]
        # scale for numerical stability
        temp[:, :] .-= logsumexp(temp)
        Ae .+= exp.(temp)
    end
    sanitize!(Ae)
    Ae ./= sum(Ae, 1)

    return Ae
end

function gen_projections(hmm::HMM, hmm_data::HMM_Data)
    K = hmm.K
    T = hmm_data.T
    S = hmm_data.S
    E_max = hmm_data.E_max

    As = AF64(K, K, E_max)

    @inbounds for e in 1:E_max
        project_A!(view(As, :, :, e), hmm, hmm_data, e)
    end

    return As
end

A_distance(As::Array{Float64,3}, i::Int, j::Int, K::Int) =
    log(K) * sum(sqrt, As[:, :, i] .* As[:, :, j])

function gen_distances(As::Array{Float64, 3}, K::Int, E_max::Int)
    D = AF64(E_max, E_max);

    for i in 1:E_max
        for j = i:E_max
            D[i, j] = A_distance(As, i, j, K)
        end
    end

    Symmetric(D)
end

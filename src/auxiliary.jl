#
# Auxiliary code
#

# makes the code cleaner and nicer
const AF64 = Array{Float64}

# a bit of overkill really with these guys
const ϵ = eps(1.0)
const log_ϵ = -1_000.0
const ϵI = UniformScaling(1e-6)

const log2π = float(Distributions.log2π)

@inline function sanitize_log!(A::AbstractArray{<:Real}, log_ϵ::Float64=log_ϵ)
    @inbounds for i in 1:length(A)
        !isfinite(A[i]) && (A[i] = log_ϵ)
    end
end

@inline function sanitize!(A::AbstractArray{<:Real}, ϵ::Float64=ϵ)
    @inbounds for i in 1:length(A)
        (!isfinite(A[i]) || A[i] ≤ ϵ) && (A[i] = ϵ)
    end
end

# blatantly copied from github.com/jwmi/HMM
"""
    logsumexp(X::AbstractArray{<:Real})

log Σᵢ exp(xᵢ) for a sequence where xᵢ = log yᵢ
∴ logsumexp(x) = log(Σᵢ yᵢ )
"""
@inline function logsumexp(X::AbstractArray{<:Real})
    mx = maximum(X)

    if !isfinite(mx)
        return mx
    else
        return mx + log(sum(exp, X .- mx))
    end
end

@inline function avg_sum(x::Vector{<:Real})
    K = Int(sqrt(length(x)))
    r = reshape(x, K, K)
    y = zeros(K)

    for i in 1:K
        for j in 1:K
            y[i] += r[i, j]
            y[j] += r[i, j]
        end
    end

    return y./2
end


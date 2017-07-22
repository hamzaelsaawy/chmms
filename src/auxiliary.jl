#
# Auxiliary code
#

# makes the code cleaner and nicer
const AF64 = Array{Float64}

const ϵ = eps(0.0)
const log_ϵ = -100_000 # just to be really sure ...
const ϵI = UniformScaling(1e-6)

log2π = float(Distributions.log2π)

@inline function sanitize_log!(A::Array{<:Real}, log_ϵ::Float64=log_ϵ)
    @inbounds for i in 1:length(A)
        !isfinite(A[i]) && (A[i] = log_ϵ)
    end
end

@inline function sanitize!(A::Array{<:Real}, ϵ::Float64=ϵ)
    @inbounds for i in 1:length(A)
        (!isfinite(A[i]) || A[i] ≤ ϵ) && (A[i] = ϵ)
    end
end

# blatantly copied from github.com/jwmi/HMM
"""
    logsumexp(x::AbstractArray{<:Real})

log Σᵢ exp(xᵢ) for a sequence where xᵢ = log yᵢ
∴ logsumexp(x) = log(Σᵢ yᵢ )
"""
@inline function logsumexp(x::AbstractArray{<:Real})
    mx = maximum(x)

    if !isfinite(mx)
        return mx
    else
        return mx + log(sum(exp(xi - mx) for xi in x))
    end
end


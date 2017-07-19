#
# Auxiliary code
#

# makes the code cleaner and nicer
const AF64 = Array{Float64}

log2π = float(Distributions.log2π)

# taken from github.com/jwmi/HMM
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


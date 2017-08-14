#
# Auxiliary code
#

"""
    empty([T::Type=Float64], dims::Int...)

Create an empty array
"""
empty(T::Type, dims::Int...) = Array{T}(dims...)
empty(dims::Int...) = empty(Float64, dims...)

# a bit of overkill really with these guys
const ϵ = eps(1.0)
const log_ϵ = -1_000.0
const ϵI = UniformScaling(1e-6)

const log2π = float(Distributions.log2π)

@inline function sanitize_log!(A::AbstractArray{<:Real}, log_ϵ::Real=log_ϵ)
    @inbounds for i in 1:length(A)
        !isfinite(A[i]) && (A[i] = log_ϵ)
    end
end

@inline function sanitize!(A::AbstractArray{<:Real}, ϵ::Real=ϵ)
    @inbounds for i in 1:length(A)
        (!isfinite(A[i]) || A[i] ≤ ϵ) && (A[i] = ϵ)
    end
end

"""
Reverse an index corresponding to (i, j) to index of (j, i)
"""
rev_ind(idx::Int, K::Int) = sub2ind((K, K), reverse(ind2sub((K, K), idx))...)

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

"""
    single_counts(x::Vector, K::Int)

Return the counts of states in [1, K] for observations in [1, K²]
"""
@inline function single_counts(x::Vector{<:Real}, K::Int)
    A = reshape(x, K, K)
    y = zeros(K)

    for i in 1:K
        for j in 1:K
            y[i] += A[i, j]
            y[j] += A[i, j]
        end
    end

    return y
end

"""
    estimate_outer(A)

Find the vector `v` that best approximates `A` with `v*v'`
`A` should be symmetric
"""
@inline function estimate_outer(A::Matrix{<:Real})
    # make symmetric
    A += A'
    l, v = eigs(A, nev=1)

    # divide by 2 b/c of adding A to its self (its transpose actually)
    return sqrt.(first(l)/2) * abs.(vec(v))
end

outer(v::AbstractVector{<:Real}, w::AbstractVector{<:Real}=v) = (v*w')[:]

"""
    outer!(A::AbstractVector{<:Real},
            v::AbstractVector{<:Real},
            w::AbstractVector{<:Real}=v)

Fill A with `v * w'`
"""
function outer!(A::AbstractMatrix{<:Real},
        v::AbstractVector{<:Real}, w::AbstractVector{<:Real}=v)
    M = length(v)
    N = length(w)

    for i in 1:M
        for j in 1:N
            A[i, j] = v[i] * w[j]
        end
    end

    return A
end


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
    get_outer(A)

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

gen_pairwise(v::AbstractVector{<:Real}) = (v*v')[:]

"""
    gen_pairwise!(A::AbstractVector{<:Real}, v::AbstractVector{<:Real})

Fill A with `v * v'`
"""
function gen_pairwise!(A::AbstractVector{<:Real}, v::AbstractVector{<:Real})
    N = length(v)
    for i in 1:N
        for j in 1:N
            idx = sub2ind((N, N), i, j)
            A[idx] = v[i] * v[j]
        end
    end

    return A
end


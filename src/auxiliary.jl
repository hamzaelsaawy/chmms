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
const ϵ = 1e-30 # << eps(1.0), so only numbers ≈ 0 will be affected
const log_ϵ = -1_000_000.0
const ϵI = UniformScaling(ϵ)

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


"""
    logsumexp(X::AbstractArray{<:Real})

log[Σᵢ exp(xᵢ)] for a sequence where xᵢ = log yᵢ
∴ logsumexp(x) = log(Σᵢ yᵢ )

Blatantly copied from github.com/jwmi/HMM
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
    pair_counts(x::Vector, K::Int)

Return the counts of states in [1, K] for observations in [1, K²], if observations from
both are being incorperated (i.e. pairwise data).

Same as sum(A, A', 1) where A = reshape(x, K, K).
"""
@inline function pair_counts(x::Vector{<:Real}, K::Int)
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
    single_counts(x::Vector, K::Int)

Return the counts of states in [1, K] for observations in [1, K²], if observations are
from the first state are being used (i.e. just states along axis 1)
"""
@inline function single_counts(x::Vector{<:Real}, K::Int)
    A = reshape(x, K, K)

    return vec(sum(A, 2))
end

"""
    estimate_outer_single(A)

Find the vector `v` that best approximates `A` with `v*v'`
`A` should be symmetric
"""
@inline function estimate_outer_single(A::Matrix{<:Real})
    # make symmetric
    A += A'
    l, v = eigs(A, nev=1)

    # divide by 2 b/c of adding A to its self (its transpose actually)
    return sqrt.(first(l)/2) * abs.(vec(v))
end

"""
    estimate_outer_double(A)

Find the vectors `u` and `v` that best approximates `A` with `u*v'`
Answer is normalized to sum to one
"""
@inline function estimate_outer_single(A::Matrix{<:Real})
    F, _ = svds(A, nsv=1)

    return _clean_svd(F.U), _clean_svd(F.Vt)
end
_clean_svd(A::AbstractArray{<:Real}) =  normalize(abs.(vec(A)), 1)

outer(v::AbstractVector{<:Real}, w::AbstractVector{<:Real}=v) = v*w'

"""
    flat_outer!(A::AbstractVector{<:Real},
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


#
# Trajectories and functions to deal with them
#

get_trajectory_range(trajptr::Vector{Int}, id::Int) =
        ( trajptr[id] ) : ( trajptr[id + 1] - 1 )

@inline function get_trajectory_from_ptr(X::Matrix{<:Real},
        trajptr::Vector{Int}, id::Int)

    return view(X, :, get_trajectory_range(trajptr, id))
end

@inline function get_pair_ranges(pairs::Matrix{Int}, id::Int)
    s1, e1, s2, e2 = pairs[:, id]

    return (s1:e1, s2:e2)
end

@inline function get_pair_from_ptr(X::Matrix{<:Real},
        pairs::Matrix{Int}, id::Int)
    r1, r2 = get_pair_ranges(pairs, id)

    return view(X, :, r1), view(X, :, r2)
end

# stolen from julia/base/sparse/sparsematrix.jl
# linear index into S.nzval
@inline function sp_sub2ind(A::SparseMatrixCSC, r::Int, c::Int)
    (1 <= r <= A.m && 1 <= c <= A.n) || return 0

    r1 = Int(A.colptr[c])
    r2 = Int(A.colptr[c+1]-1)

    (r1 > r2) && return 0
    r1 = searchsortedfirst(A.rowval, r, r1, r2, Base.Order.Forward)
    ((r1 > r2) || (A.rowval[r1] != r)) ? 0 : r1
end


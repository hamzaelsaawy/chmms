#
# Trajectories and functions to deal with them
#

#=
struct Trajectories
    X::Matrix{Float64} # D × number of observations
    traj_ptr::Vector{Int} # see SparseMatrixCSC
    traj_pairs::Vector{NTuple{2, Int}} # id of pairs
    traj_singles::Vector{Int} # id of single trajs
end

num_obs(T::Trajectories) = size(T.X, 2)
num_trajs(T::Trajectories) = length(T.traj_ptr) - 1
num_pairs(T::Trajectories) = length(T.traj_pairs)
num_singles(T::Trajectories) = length(T.traj_singles)
traj_lengths(T::Trajectories) = diff(T.traj_ptr)

@inline function get_trajectory(id::Int, T::Trajectories)
    start_traj = T.traj_ptr[id]
    end_traj = T.traj_ptr[id + 1] - 1

    return view(T.X, :, start_traj:end_traj)
end
=#

@inline function get_trajectory_from_ptr(id::Int,
        X::Matrix{Float64}, traj_ptr::Vector{Int})
    start_traj = traj_ptr[id]
    end_traj = traj_ptr[id + 1] - 1

    return view(X, :, start_traj:end_traj)
end

# stolen from julia/base/sparse/sparsematrix.jl
@inline function sp_sub2ind(A::SparseMatrixCSC, r::Int, c::Int)
    (1 <= r <= A.m && 1 <= c <= A.n) || return 0

    r1 = Int(A.colptr[c])
    r2 = Int(A.colptr[c+1]-1)

    (r1 > r2) && return 0
    r1 = searchsortedfirst(A.rowval, r, r1, r2, Base.Order.Forward)
    ((r1 > r2) || (A.rowval[r1] != r)) ? 0 : r1
end

@inline function get_trajectory_from_frame(S::SparseTrajData, X::Matrix{Float64},
        c::Int, start_frame::Int, end_frame::Int)
    s = sp_sub2ind(S, start_frame, c)
    e = sp_sub2ind(S, end_frame, c)

    return view(X, :, s:e)
end


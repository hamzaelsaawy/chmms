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

@inline function get_trajectory(id::Int, X::Matrix{Float64}, traj_ptr::Vector{Int})
    start_traj = traj_ptr[id]
    end_traj = traj_ptr[id + 1] - 1

    return view(X, :, start_traj:end_traj)
end


#
# Converting Trajdata to SparseMatrix, and mapping functions on trajectories
#

const SparseTrajData = SparseMatrixCSC{AutomotiveDrivingModels.VehicleState, Int}

"""
    td_sparse(td::Trajdata)

Returns a sparse array and a dictionary.

In the sparse array, each column is a car; row, a frame/scene;
entry, `AutomotiveDrivingModels.VehicleState`.
Also returns a lookup table (`Dict`) of ids (`Records.RecordState.id`) to
index (column number).
"""
function td_sparse(td::Trajdata)
    # is this an efficient data pipeline?
    # probably not
    # do we care?
    # probably not

    # ids are by time of entry
    # translate id to index on range of 1:n
    id_lookup = Dict(id => index for (index, id) in
        (td.defs |> keys |> collect |> sort |> enumerate))

    m = nframes(td) # num rows
    n = nids(td) # num cols
    n_st = length(td.states)

    # the row/frame of each state
    I = Vector{Int}(n_st)
    # the column/car index (not the id, see id_lookup)
    J = similar(I)
    # the states themselves
    V = similar(I, VehicleState)

    # the index into I, J, and V (on [1:n_st])
    idx = 1
    for (fid, frame) in enumerate(td.frames)
        for stateid in frame.lo : frame.hi
            recstate = td.states[stateid]
            I[idx] = fid
            J[idx] = id_lookup[recstate.id]
            V[idx] = recstate.state

            idx += 1
        end
    end

    S = Base.sparse(I, J, V, m, n)
    return (S, id_lookup)
end

################################################################################
# Data Extraction
################################################################################
"""
    traj_nested(S::SparseTrajData, fs...)

Return a vector of trajectories, where each trajectory is a matrix with
the first column containing the seconds into the dataset (`frame * td.timestep`)
and the subsequent containing `f(s::VehicleState)::Float64` for `f ∈ fs`.
"""
function traj_nested(S::SparseTrajData, fs...)
    # each array is the scene number and f(state) for nonzero entries of S
    trajs = Vector{Matrix{Float64}}(S.n);

    for id in 1:S.n
        # range of values in rowvals (& nzval) that belong to the colomn/car `id`
        rng = nzrange(S, id)
        offset = first(rng) - 1
        traj = AF64(length(rng), length(fs)+1)

        for i in rng
            traj[i - offset, 1] = S.rowval[i]

            for (j, f) in fs |> enumerate
                traj[i - offset, 1 + j] = f(S.nzval[i])
            end
        end

        trajs[id] = traj
    end

    return trajs
end


"""
    traj_flat(td::SparseTrajData, T, fs...)

Return a matrix `D[i, t] = fs[i](S.nzval[t])`
"""
function traj_flat(S::SparseTrajData, fs::Vector{Function}, T=1_000_000,)
    T = min(5000, nnz(S))
    Y = AF64(length(fs), T)

    for t in 1:T
        s = S.nzval[t]

        for (i, f) in fs |> enumerate
            @inbounds Y[i, t] = f(s)
        end
    end

    return Y
end


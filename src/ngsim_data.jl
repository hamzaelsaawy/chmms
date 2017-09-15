#
# Converting Trajdata to SparseMatrix, and mapping functions on trajectories
#

################################################################################
# Convert to SparseMatrix (easier to use)
################################################################################

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
@inline function traj_flat(S::SparseTrajData, fs::Vector{Function}, T::Int=10_000, )
    (T > nnz(S)) && error("T > nnz(S)")
    fs_enum = enumerate(fs)
    Y = AF64(length(fs), T)

    @inbounds for t in 1:T
        s = S.nzval[t]

        for (i, f) in fs_enum
            Y[i, t] = f(s)
        end
    end

    return Y
end

################################################################################
# Finding (allegedly interacting) pairs in code
################################################################################

"""
    find_pairs(S::SparseTrajData; lane_max::Int=1, s_max::Real=10,
            max_interaction::Int=300, min_length::Int=20)

Return a Dict of tuple of column numbers (car ids) to the frames that the cars
are (possibly) interacting over.

`lane_max`: how many lanes away a car can be (lanes are 3.7m/12ft wide)  
`s_max`: max distance along a lane two cars can be (meters) TODO  
`max_interaction`: number of cars/columns away to search.
trajectories are (more or less) sequential in time of arrival
assume that 300 cars latter, the cars are in different time periods  
`min_length`: length of interaction (frames) to be considered significant and stored
1 frame is 0.1 seconds
"""
function find_pairs(S::SparseTrajData; lane_max::Int=1, s_max::Real=10,
        max_interaction::Int=300, min_length::Int=20)
    # does this even do anything?
    s_max = float(s_max)

    start_iter = 0
    end_iter = 0
    si = VehicleState()
    sj = VehicleState()

    # maps two cars to the frames they "interact" in
    pair_traj = Dict{NTuple{2, Int}, NTuple{2, Int}}()

    for i in 1:(S.n-1)
        max_j = min(i + max_interaction, S.n)

        for j in (i + 1):(max_j)
            frames_i = view(S.rowval, nzrange(S, i))
            frames_j = view(S.rowval, nzrange(S, j))

            # first frame (time step) that both trajectories are defined
            start_frame = maximum(map(first, [frames_i, frames_j]))
            # earliest frame (time step) that either trajectory ends
            end_frame = minimum(map(last, [frames_i, frames_j]))

            rng = start_frame:end_frame

            if length(rng) < min_length
                continue
            end

            start_iter = 0
            for (k, fr) in enumerate(rng)
                si = S[fr, i]
                sj = S[fr, j]

                # TODO change this to posF.s stuff...
                ds = abs(si.posG - sj.posG)
                dlane = abs(si.posF.roadind.tag.lane -
                    sj.posF.roadind.tag.lane)

                # cars interacting?
                if (ds <= s_max) && (dlane <= lane_max)
                    (start_iter == 0) && (start_iter = fr)
                    end_iter = fr
                end
            end

            if (start_iter != 0) && ( (end_iter - start_iter + 1) >= min_length )
                pair_traj[(i, j)] = (start_iter, end_iter)
            end
        end
    end

    return pair_traj
end


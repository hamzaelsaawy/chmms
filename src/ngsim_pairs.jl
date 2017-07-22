#
# Finding (allegedly interacting) pairs in code
#


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


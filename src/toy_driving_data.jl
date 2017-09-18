#
# Generate 1d Driving Data
#
using AutomotiveDrivingModels
include("trajectories.jl")

"""
    gen_simple_traj(; road_length::Int=1200, max_starting_pos::Int=400,
        n_cars::Int=5, timestep::Float64=0.1, avg_v::Float64=10.0, std_v::Float64=1.0,
        n_ticks::Int=floor(Int, (road_length-max_starting_pos) / avg_v / timestep),
    )

Returns an Array{Float64, 3} of size (n_cars, n_ticks, 2), where the last dimension
is the position and speed.
"""
function gen_simple_traj(;
        road_length::Real=1200, # [m]
        max_starting_pos::Real=400,
        n_cars::Integer=5,
        timestep::Real=0.1,
        avg_v::Real=10.0, # [m/s]
        std_v::Real=1.0,
        min_v::Real=3.0,
        n_ticks::Integer=floor(Int, (road_length-max_starting_pos) / avg_v / timestep),
    )
    # cars can overlap ... oh well

    roadway = StraightRoadway(road_length)
    scene = Scene1D(n_cars)
    models = Dict{Int,LaneFollowingDriver}()

    vd = VehicleDef()

    # first car does not accelerate, goes the average speed, and reaches the end first
    push!(scene, Entity(State1D(max_starting_pos, max(avg_v, min_v)), vd, 1))
    models[1] = StaticLaneFollowingDriver(0.0)

    for i in 2:n_cars
        s = rand() * (max_starting_pos - vd.length)
        v = max(min_v, avg_v + rand() * std_v)
        state = State1D(s, v)
        push!(scene, Entity(State1D(s, v), vd, i))
        models[i] = IntelligentDriverModel(v_des=(avg_v + randn() * std_v))
    end

    rec = QueueRecord(Vehicle1D, n_ticks, timestep)
    simulate!(LaneFollowingAccel, rec, scene, roadway, models, n_ticks)

    D = Array{Float64}(n_cars, n_ticks, 2)

    for t in 1:n_ticks
        f = rec.frames[n_ticks - t + 1]
        for c in 1:scene.n
            s = f.entities[c].state
            D[c, t, 1] = s.s
            D[c, t, 2] = s.v
        end
    end

    return D
end

function gen_mult_trajs(n_scenes::Integer=500;
        timestep::Real=0.1,
        road_lengths::AbstractVector=1_000:1_500,
        n_cars::AbstractVector=3:6,
        avg_vs::AbstractVector=[10.0],
        std_vs::AbstractVector=[1.0],
        max_starting_pos::Real=400,)
    scenes = [gen_simple_traj(
                road_length=rand(road_lengths),
                n_cars=rand(n_cars),
                avg_v=rand(avg_vs),
                std_v=rand(std_vs),
                ) for _ in 1:n_scenes]
    n_obs = sum(prod(size(s, 1, 2)) for s in scenes)
    n_trajs = sum(size(s, 1) for s in scenes)

    X = zeros(3, n_obs)
    trajptr = zeros(Int, n_trajs + 1)

    loc_X = 0
    loc_traj = 1
    for s in scenes
        n_cars, T, _ = size(s)

        for c in 1:n_cars
            r = loc_X + (1:T)
            X[1:2, r] = s[c, :, :]'

            trajptr[loc_traj] = first(r)
            trajptr[loc_traj + 1] = last(r)

            loc_X += T
            loc_traj += 1
        end
    end
    trajptr[end] += 1

    n_pairs = sum(choose2(size(s, 1)) for s in scenes)
    pairs = zeros(Int, 4, n_pairs)

    loc_pairs = 1
    loc_traj = 1
    for s in scenes
        n_cars = size(s, 1)
        ptrs = trajptr[loc_traj + (0:n_cars)]
        rs = map(colon, ptrs[1:(end - 1)], ptrs[2:end] .- 1)

        for i in 1:n_cars
            for j in 1:(i - 1)
                pairs[1, loc_pairs] = first(rs[i])
                pairs[2, loc_pairs] = last(rs[i])
                pairs[3, loc_pairs] = first(rs[j])
                pairs[4, loc_pairs] = last(rs[j])

                loc_pairs += 1
            end
        end

        loc_traj += n_cars
    end

    Δt = timestep
    for i in 1:n_trajs
        r = get_trajectory_range(trajptr, i)
        T = length(r)
        Vt = view(X, 2, r)
        At = view(X, 3, r)

        # forward diff, Δₕ x
        At[1] = (Vt[2] - Vt[1]) / Δt
        # central diff, δₕ x
        At[2:(T - 1)] = (Vt[3:T] .- Vt[1:(T - 2)]) ./ (2 * Δt)
        # backward diff, ∇ₕ
        At[T] = (Vt[T] - Vt[T - 1]) / Δt
    end

    return X, trajptr, pairs
end

choose2(n::Int) = n * (n - 1) / 2

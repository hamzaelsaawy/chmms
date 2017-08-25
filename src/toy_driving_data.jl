#
# Generate 1d Driving Data
#
using AutomotiveDrivingModels

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
        n_cars::Int=5,
        timestep::Float64=0.1,
        avg_v::Real=10.0, # [m/s]
        std_v::Real=1.0,
        n_ticks::Int=floor(Int, (road_length-max_starting_pos) / avg_v / timestep),
    )
    # cars can overlap ... oh well

    roadway = StraightRoadway(road_length)
    scene = Scene1D(n_cars)
    models = Dict{Int, LaneFollowingDriver}()

    vd = VehicleDef()

    # first car does not accelerate, goes the average speed, and reaches the end first
    push!(scene, Entity(State1D(max_starting_pos, avg_v), vd, 1))
    models[1] = StaticLaneFollowingDriver(0.0)

    for i in 2:n_cars
        s = rand() * (max_starting_pos - vd.length)
#        floor(rand() * (max_starting_pos - vd.length) / vd.length + 1) * vd.length
        v = avg_v + randn() * std_v
        state = State1D(s, v)
        push!(scene, Entity(State1D(s, v), vd, i))
        models[i] = IntelligentDriverModel(v_des=(avg_v + randn()*std_v))
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


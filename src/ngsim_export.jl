#
# Exporting NGSIM data to friendlier (& more lightweight) format
#

"""
    export_ngsim_traj(traj_num::Int, data_path::AbstractString="../data/")

Saves the NGSIM data in a more friendly format.

(Dumps most of the data).
Creates a new folder in named `$(data_path)/$(ngsim_file_name)`
Saves the trajectory data object from `NGSIM.jl` in a JLD2 format
Saves the (sequential) car id, lane, velocity, heading, and acceleration
    (estiamted by finite-differece) as `X.csv`
Saves the `trajectory pointer` (see MatrixCSC.colptr) as `traj_ptr.csv`
Saves the pairs start and stop (similar to trajectory pointers) as `pairs.csv`
Saves the timestep (seconds) in `time_step.csv`
"""
function export_ngsim_traj(traj_num::Int, data_path::AbstractString="../data/")
    traj_file = Base.Filesystem.basename(NGSIM.TRAJDATA_PATHS[traj_num])
    # remove the "trajectory_" and ".csv"
    traj_file = traj_file[10:search(traj_file, '.')-1]
    data_path = joinpath(data_path, traj_file)

    if isdir(data_path)
        info("Saving data to `$(data_path)`.")
    else
        Base.Filesystem.mkdir(data_path)
        info("Created directory `$(data_path)`.")
    end


    td = load_trajdata(traj_num)
    JLD.@save joinpath(data_path, "td.jld") td
    info("Saving trajectory object to `td.jld`.")

    TS = td_sparse(td)

    if contains(traj_file, "i80")
        rd = NGSIM.ROADWAY_80
    elseif contains(traj_file, "i101")
        rd = NGSIM.ROADWAY_101
    else
        error("unknown roadway")
    end

   fs = [
        ("x", s::VehicleState -> s.posG.x),
        ("y", s::VehicleState -> s.posG.y),
        ("heading", s::VehicleState -> s.posF.ϕ),
        ("velocity", s::VehicleState -> s.v),
        ("lane", s::VehicleState ->
            length(rd[s.posF.roadind.tag.segment].lanes) - s.posF.roadind.tag.lane + 1)
    ]

    X = Array{Float64}(nnz(TS), length(fs))
    @inbounds for (t, s) in TS.nzval |> enumerate
        for (i, (_, f)) in enumerate(fs)
            X[t, i] = f(s)
        end
    end

    # make a dataframe from the extracted data, and then give them names
    df = DataFrame([X TS.rowval])
    names!(df, vcat(map(Symbol ∘ first, fs), :frame))

    # make frame and lane Ints
    for s in [:frame, :lane]
        df[s] = Int.(df[s])
    end

    # estimate acceleration
    Δt = td.timestep
    V = df[:velocity]
    A = zeros(V)
    num_traj = TS.n

    for i in 1:num_traj
        r = get_trajectory_range(TS.colptr, i)
        T = length(r)
        Vt = view(V, r)
        At = view(A, r)

        # forward diff, Δₕ x
        At[1] = (Vt[2] - Vt[1]) / Δt
        # central diff, δₕ x
        At[2:(T-1)] = (Vt[3:T] .- Vt[1:(T-2)]) ./ (2*Δt)
        # backward diff, ∇ₕ
        At[T] = (Vt[T] - Vt[T-1]) / Δt
    end

    df[:acceleration] = A

    # the car id
    car = zeros(Int, length(V))
    for i in 1:num_traj
        r = get_trajectory_range(TS.colptr, i)
        car[r] = i
    end

    CSV.write(joinpath(data_path, "X.csv"), df, header=true)
    info("Saving trajectory csv to `X.csv`.")

    writecsv(joinpath(data_path, "traj_ptr.csv"), TS.colptr)
    info("Saving trajectory pointer to `traj_pts.csv`.")

    writecsv(joinpath(data_path, "time_step.csv", td.timestep)
    info("Saving trajectory pointer to `time_step.csv`.")

#Saves the pairs start and stop (similar to trajectory pointers) as `pairs.csv`
end

#
# Main file for NGSIM-related code
#

import Distributions
import StatsBase: wsample
import Clustering: kmeans
#import Base: @propagate_inbounds
using AutomotiveDrivingModels
using NGSIM

include("auxiliary.jl")
include("ngsim_data.jl")
include("trajectories.jl")
include("chmm_main.jl")
include("chmm_training.jl")


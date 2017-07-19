#
# Main file for NGSIM-related code
#

using AutomotiveDrivingModels
using NGSIM
import Distributions
import Base: @propagate_inbounds

include("auxiliary.jl")
include("ngsim_data.jl")
include("ngsim_pairs.jl")
include("ngsim_em_init.jl")
include("ngsim_em.jl")


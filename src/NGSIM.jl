#
# Main file for NGSIM-related code
#

using AutomotiveDrivingModels
using NGSIM
import Distributions
import Base: @propagate_inbounds

include("auxiliary.jl")
include("ngsim_data.jl")
#include("em_init.jl")
#include("em.jl")
#include("similarity.jl")


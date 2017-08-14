#
# Pseudo-module for CHMM stuff
#

import StatsBase: wsample
import Clustering: kmeans

include("auxiliary.jl")
include("trajectories.jl")
include("chmm_main.jl")
include("chmm_training.jl")


#
# Main file for CHMM-related code
#

# todo:
#  tests
#  switch to static arrays

module CHMM
    import Distributions: MvNormal, logpdf
    import StatsBase: wsample
    import Clustering: kmeans

    include("auxiliary.jl")
    include("trajectories.jl")
    include("chmm_main.jl")
    include("chmm_training.jl")
end

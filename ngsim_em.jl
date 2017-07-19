#
# EM training procedure
#

@inline function logmvnormal(hmm::HMM, hmm_data::HMM_Data,
        m::Int, k::Int, t::Int)
    # shamelessly stolen from Distributions.jl/src/multivariate/mvnormal.jl
    lpdf = -(hmm.NΓ * log2π  + hmm_data.logdetΣs[m, k])/2
    o = hmm_data.YΓ[:, t] - hmm.μs[:, m, k]
    lpdf -= dot(o, (hmm_data.invΣs[:, :, m, k] * o)) / 2

    return lpdf
end


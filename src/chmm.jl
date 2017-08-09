#
# CHMM generation and stuff...
#

function rand_chmm(K, D)
    π0 = rand(K)
    π0 ./= sum(π0)

    # p(zᵗ⁺¹ | z₁ᵗ, z₂ᵗ) = P[zᵗ⁺¹, z₁ᵗ, z₂ᵗ]
    # Z₁ ≡ Z₂
    P = rand(K, K, K)
    P ./= sum(P, 1)

    μs = [randn(D) * 8 for _ in 1:K]
    Σs = [eye(D) * rand() for _ in 1:K]

    model = Dict(:π0 => π0, :P => P,
        :μs => μs, :Σs => Σs, :K => K, :D => D)

    return model
end

function simulate_model(model, T)
    D = model[:D]
    Z = Matrix{Int}(2, T)
    X1 = Matrix{Float64}(D, T)
    X2 = similar(X1)

    sqrtm_Σs = map(sqrtm, model[:Σs])
    z = wsample(1:K, model[:π0], 2)

    for t in 1:T
        z = wsample(1:K, model[:P][:, z...], 2)
        Z[:, t] = z
        X1[:, t] = sqrtm_Σs[z[1]] * randn(D) + model[:μs][z[1]]
        X2[:, t] = sqrtm_Σs[z[2]] * randn(D) + model[:μs][z[2]]
    end

    return (Z, X1, X2)
end


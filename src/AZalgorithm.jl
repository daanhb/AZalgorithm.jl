module AZalgorithm

using LinearAlgebra, LinearMaps, LowRankApprox

export az,
    tsvd

include("rank.jl")
include("az.jl")

end # module AZalgorithm

module AZalgorithm

using LinearAlgebra, LinearMaps, LowRankApprox

export az,
    az_factorize,
    tsvd

include("tsvd.jl")
include("rank.jl")
include("az.jl")

end # module AZalgorithm

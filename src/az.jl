
"A tolerance for use in the AZ algorithm."
az_tolerance(x) = az_tolerance(eltype(x))
az_tolerance(::Type{Complex{T}}) where T = az_tolerance(T)
az_tolerance(::Type{T}) where {T <: AbstractFloat} = 100eps(T)

"Construct the AZ operator `A - AZ'A` in operator form."
az_AminAZA(A::AbstractMatrix, Zstar::AbstractMatrix) =
    A - A*Zstar*LinearMaps.LinearMap(A)

"Construct the AZ operator `I - AZ'` in operator form."
az_IminAZ(A::AbstractMatrix, Zstar::AbstractMatrix) =
    I - A*LinearMaps.LinearMap(Zstar)

# if the given operators are not matrices, we assume that they know how
# to compose themselves with other operators
az_AminAZA(A, Zstar) = A - A*Zstar*A
az_IminAZ(A, Zstar) = I - A*Zstar


"Storage type for all operators involved in the AZ algorithm."
abstract type AZFactorization end

"Factorization obtained by LowRankApprox.jl."
struct LRA_AZFact <: AZFactorization
    A                   # the A operator
    Zstar               # the Z' operator
    AminAZA             # A - AZ'A
    IminAZ              # I - AZ'
    az1_fact            # low-rank factorization of AminAZA
end

"Factorization obtained by LowRankApprox.jl."
struct RandCol_AZFact <: AZFactorization
    A                   # the A operator
    Zstar               # the Z' operator
    AminAZA             # A - AZ'A
    IminAZ              # I - AZ'
    R                   # random matrix
    col_fact            # factorization of column space of AminAZA
end

# apply AZ to a vector
Base.:*(F::LRA_AZFact, b) = az_lra(b, F.A, F.Zstar, F.az1_fact, F.IminAZ)
Base.:*(F::RandCol_AZFact, b) =
    az_randcol(b, F.A, F.Zstar, F.R, F.col_fact, F.IminAZ)

function az_lra(b, A, Zstar, az1_fact, IminAZ)
    x1 = az1_fact \ (IminAZ*b)  # step 1
    x2 = Zstar * (b - A*x1)     # step 2
    x1+x2                       # step 3
end

function az_randcol(b, A, Zstar, R, col_fact, IminAZ)
    u1 = col_fact \ (IminAZ*b)
    x1 = R * u1                 # step 1
    x2 = Zstar * (b - A*x1)     # step 2
    x1+x2                       # step 3
end

"""
    az(A, Zstar, b)

Compute an AZ factorization with the given `(A,Z)` pair.

The function accepts the same options as the `az` function.
"""
function az_factorize(A, Zstar;
        atol            = 0.0,
        rtol            = az_tolerance(A),
        method          = :lra,
        rank            = method == :lra ? -1 : 30,
        verbose         = false)

    AminAZA = az_AminAZA(A, Zstar)
    IminAZ = az_IminAZ(A, Zstar)
    if method == :lra
        if rank > 0     # use fixed maximal rank
            az1_fact = psvdfact(AminAZA; atol, rtol, rank)
        else            # let psvdfact determine rank adaptively
            az1_fact = psvdfact(AminAZA; atol, rtol)
        end
        verbose && println("AZ factorization completed with rank: $(length(az1_fact.:S))")
        LRA_AZFact(A, Zstar, AminAZA, IminAZ, az1_fact)
    elseif method == :randcol
        R, Col = randomized_column_space(AminAZA, rank)
        col_fact = tsvd_factorize(Col, atol, rtol)
        RandCol_AZFact(A, Zstar, AminAZA, IminAZ, R, col_fact)
    else
        error("Unkown factorization method: $(method)")
    end
end

"""
    az(A, Zstar, b)

Apply the AZ algorithm to the right hand side vector `b` with the given
combination of `A` and `Z`. The matrix `Z` is given in its adjoint form `Z'`.
"""
function az(A, Zstar, b; options...)
    fact = az_factorize(A, Zstar; options...)
    fact * b
end

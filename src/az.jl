
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

"Factorization of the AZ algorithm for enriched basis using the inverse of A11 (A = [A11 A12; A21 A22])."
struct EnrichedInv_AZFact <: AZFactorization
    A12                 # the A12 operator
    A21                 # the A21 operator
    A22                 # the A22 operator
    Z11star             # the Z11' operator
    AminAZA             # A - AZ'A
end

"Factorization of the AZ algorithm for enriched basis using the left inverse of A11 (A = [A11 A12; A21 A22])."
struct EnrichedLInv_AZFact <: AZFactorization
    A11                 # the A11 operator
    A12                 # the A12 operator
    A21                 # the A21 operator
    A22                 # the A22 operator
    Z11star             # the Z11' operator
    AminAZA             # factorization of A - AZ'A
end


"Factorization of the AZ algorithm for enriched basis using the left inverse of A11 (A = [A11 A12])."
struct EnrichedLInv_reduced_AZFact <: AZFactorization
    A11                 # the A11 operator
    A12                 # the A12 operator
    Z11star             # the Z11' operator
    AminAZA             # factorization of A - AZ'A
end

# apply AZ to a vector
Base.:*(F::LRA_AZFact, b) = az_lra(b, F.A, F.Zstar, F.az1_fact, F.IminAZ)
Base.:*(F::RandCol_AZFact, b) =
    az_randcol(b, F.A, F.Zstar, F.R, F.col_fact, F.IminAZ)
Base.:*(F::EnrichedInv_AZFact, b) = 
    az_enrichedinv(b[1], b[2], F.A12, F.A21, F.Z11star, F.AminAZA)
Base.:*(F::EnrichedLInv_AZFact, b) = 
    az_enrichedlinv(b[1], b[2], F.A11, F.A12, F.A21, F.Z11star, F.AminAZA)
Base.:*(F::EnrichedLInv_reduced_AZFact, b) = 
    az_enrichedlinv_reduced(b, F.A11, F.A12, F.Z11star, F.AminAZA)

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

function az_enrichedinv(b1, b2, A12, A21, Z11star, AminAZA)
    x1 = AminAZA \ (b2 - A21*Z11star*b1)    # step 1
    x2 = Z11star * (b1 - A12*x1)            # step 2
    (x2,x1)                                 # step 3
end

function az_enrichedlinv(b1, b2, A11, A12, A21, Z11star, AminAZA)
    x1 = AminAZA \ [b1 - A11*Z11star*b1; b2 - A21*Z11star*b1]       # step 1
    x2 = Z11star * (b1 - A12*x1)                                    # step 2
    (x2,x1)                                                         # step 3
end

function az_enrichedlinv_reduced(b, A11, A12, Z11star, AminAZA)
    x1 = AminAZA \ (b - A11*Z11star*b)      # step 1
    x2 = Z11star * (b - A12*x1)             # step 2
    (x2,x1)                                 # step 3
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
    enrichedaz_factorize(A, Zstar)

"""
function enrichedaz_factorize(Z11star, A...; 
    atol            = 0.0,
    rtol            = az_tolerance(A[1]),
    verbose         = false)

    if(length(A) == 2)
        AminAZA = tsvd_factorize(Matrix(A[2] - A[1]*Z11star*A[2]), atol, rtol)
        EnrichedLInv_reduced_AZFact(A[1], A[2], Z11star, AminAZA)
    elseif(length(A) == 3)
        AminAZA = tsvd_factorize(Matrix(A[3] - A[2]*Z11star*A[1]), atol, rtol)
        EnrichedInv_AZFact(A[1], A[2], A[3], Z11star, AminAZA)
    elseif(length(A) == 4)
        AminAZA = tsvd_factorize([Matrix(A[2] - A[1]*Z11star*A[2]); Matrix(A[4] - A[3]*Z11star*A[2])], atol, rtol)
        EnrichedLInv_AZFact(A[1], A[2], A[3], A[4], Z11star, AminAZA)
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

"""
    az(A12, A21, A22, Z11star, b1, b2)

Apply the AZ algorithm for enriched bases to the right hand side vector `b` = [`b1`; `b2`] with a 
combination of `A` = [`A11` `A12`; `A21` `A22`] and `Z` = [`Z11` 0; 0 0]. The matrix `Z11` is 
given in its adjoint form `Z11'`. The adjoint matrix `Z11'` is the inverse of the matrix `A11`.
"""
function az(A12, A21, A22, Z11star, b1, b2; options...)
    fact = enrichedaz_factorize(Z11star, A12, A21, A22; options...)
    fact * (b1, b2)
end

"""
    az(A11, A12, Z11star, b)

Apply the AZ algorithm for enriched bases to the right hand side vector `b` with a 
combination of `A` = [`A11` `A12`] and `Z` = [`Z11` 0]. The matrix `Z11` is 
given in its adjoint form `Z11'`. The adjoint matrix `Z11'` is the left inverse of the matrix `A11`.
"""
function az(A11, A12, Z11star, b; options...)
    fact = enrichedaz_factorize(Z11star, A11, A12; options...)
    fact * b
end

"""
    az(A11, A12, A21, A22, Z11star, b1, b2)

Apply the AZ algorithm for enriched bases to the right hand side vector `b` = [`b1`; `b2`] with a 
combination of `A` = [`A11` `A12`; `A21` `A22`] and `Z` = [`Z11` 0; 0 0]. The matrix `Z11` is 
given in its adjoint form `Z11'`. The adjoint matrix `Z11'` is the left inverse of the matrix `A11`.
"""
function az(A11, A12, A21, A22, Z11star, b1, b2; options...)
    fact = enrichedaz_factorize(Z11star, A11, A12, A21, A22; options...)
    fact * (b1, b2)
end
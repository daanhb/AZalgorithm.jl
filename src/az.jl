
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

"Construct the subblock (A - AZ'A)12"
function enrichedaz_AminAZA_12(A11, A12, Z11star)
    MN = size(A12,1)
    K = size(A12,2)
    AminAZA = zeros(eltype(A12), MN, K)
    for i in 1:K
        col = A12*I[1:K,i]
        AminAZA[:,i] = (I - A11*Z11star)*col
    end
    AminAZA
end

"Construct the subblock (A - AZ'A)22"
function enrichedaz_AminAZA_22(A12, A21, A22, Z11star)
    MK = size(A22,1)
    K = size(A22,2)
    AminAZA = zeros(eltype(A22), MK, K)
    for i in 1:K
        AminAZA[:,i] = (A22-A21*Z11star*A12)*I[1:K,i]
    end
    AminAZA
end

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
struct Inv_EnrichedAZFact <: AZFactorization
    A12                 # the A12 operator
    A21                 # the A21 operator
    A22                 # the A22 operator
    Z11star             # the Z11' operator
    AminAZA             # A - AZ'A
end

"Factorization of the AZ algorithm for enriched basis using the left inverse of A11 (A = [A11 A12; A21 A22])."
struct Linv_EnrichedAZFact <: AZFactorization
    A11                 # the A11 operator
    A12                 # the A12 operator
    A21                 # the A21 operator
    A22                 # the A22 operator
    Z11star             # the Z11' operator
    AminAZA             # factorization of A - AZ'A
end


"Factorization of the AZ algorithm for enriched basis using the left inverse of A11 (A = [A11 A12])."
struct Linvred_EnrichedAZFact <: AZFactorization
    A11                 # the A11 operator
    A12                 # the A12 operator
    Z11star             # the Z11' operator
    AminAZA             # factorization of A - AZ'A
end

# apply AZ to a vector
Base.:*(F::LRA_AZFact, b) = az_lra(b, F.A, F.Zstar, F.az1_fact, F.IminAZ)
Base.:*(F::RandCol_AZFact, b) =
    az_randcol(b, F.A, F.Zstar, F.R, F.col_fact, F.IminAZ)
Base.:*(F::Inv_EnrichedAZFact, b) = 
    enrichedaz_inv(b, F.A12, F.A21, F.Z11star, F.AminAZA)
Base.:*(F::Linv_EnrichedAZFact, b) = 
    enrichedaz_linv(b, F.A11, F.A12, F.A21, F.Z11star, F.AminAZA)
Base.:*(F::Linvred_EnrichedAZFact, b) = 
    enrichedaz_linvred(b, F.A11, F.A12, F.Z11star, F.AminAZA)

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

function enrichedaz_inv(b, A12, A21, Z11star, AminAZA)
    MN = size(Z11star,2)
    x1 = AminAZA \ (b[MN+1:end] - A21*Z11star*b[1:MN])    # step 1
    x2 = Z11star * (b[1:MN] - A12*x1)                     # step 2
    [x2; x1]                                              # step 3
end

function enrichedaz_linv(b, A11, A12, A21, Z11star, AminAZA)
    MN = size(Z11star,2)
    x1 = AminAZA \ [(I - A11*Z11star)*b[1:MN]; b[MN+1:end] - A21*Z11star*b[1:MN]]       # step 1
    x2 = Z11star * (b[1:MN] - A12*x1)                                                   # step 2
    [x2; x1]                                                                            # step 3
end

function enrichedaz_linvred(b, A11, A12, Z11star, AminAZA)
    x1 = AminAZA \ (b - A11*Z11star*b)      # step 1
    x2 = Z11star * (b - A12*x1)             # step 2
    [x2; x1]                                # step 3
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
    enrichedazinv_factorize(A12, A21, A22, Z11star)

"""
function enrichedazinv_factorize(A12, A21, A22, Z11star; 
    atol            = 0.0,
    rtol            = az_tolerance(A12),
    verbose         = false)

    AminAZA = tsvd_factorize(enrichedaz_AminAZA_22(A12, A21, A22, Z11star), atol, rtol)
    Inv_EnrichedAZFact(A12, A21, A22, Z11star, AminAZA)
end

"""
    enrichedazlinv_factorize(A11, A12, A21, A22, Z11star)

"""
function enrichedazlinv_factorize(A11, A12, A21, A22, Z11star; 
    atol            = 0.0,
    rtol            = az_tolerance(A11),
    verbose         = false)

    AminAZA = tsvd_factorize([enrichedaz_AminAZA_12(A11, A12, Z11star); enrichedaz_AminAZA_22(A12, A21, A22, Z11star)], atol, rtol)
    Linv_EnrichedAZFact(A11, A12, A21, A22, Z11star, AminAZA)
end

"""
    enrichedazlinvred_factorize(A11, A12, Z11star)

"""
function enrichedazlinvred_factorize(A11, A12, Z11star; 
    atol            = 0.0,
    rtol            = az_tolerance(A11),
    verbose         = false)

    AminAZA = tsvd_factorize(enrichedaz_AminAZA_12(A11, A12, Z11star), atol, rtol)
    Linvred_EnrichedAZFact(A11, A12, Z11star, AminAZA)
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
    enrichedaz(A12, A21, A22, Z11star, b)

Apply the AZ algorithm for enriched bases to the right hand side vector `b` with a 
combination of `A` = [`A11` `A12`; `A21` `A22`] and `Z` = [`Z11` 0; 0 0]. The matrix `Z11` is 
given in its adjoint form `Z11'`. The adjoint matrix `Z11'` is the inverse of the matrix `A11`.
"""
function enrichedaz(A12, A21, A22, Z11star, b; options...)
    fact = enrichedazinv_factorize(A12, A21, A22, Z11star; options...)
    fact * b
end

"""
    enrichedaz(A11, A12, Z11star, b)

Apply the AZ algorithm for enriched bases to the right hand side vector `b` with a 
combination of `A` = [`A11` `A12`] and `Z` = [`Z11` 0]. The matrix `Z11` is 
given in its adjoint form `Z11'`. The adjoint matrix `Z11'` is the left inverse of the matrix `A11`.
"""
function enrichedaz(A11, A12, Z11star, b; options...)
    fact = enrichedazlinvred_factorize(A11, A12, Z11star; options...)
    fact * b
end

"""
    enrichedaz(A11, A12, A21, A22, Z11star, b)

Apply the AZ algorithm for enriched bases to the right hand side vector `b` with a 
combination of `A` = [`A11` `A12`; `A21` `A22`] and `Z` = [`Z11` 0; 0 0]. The matrix `Z11` is 
given in its adjoint form `Z11'`. The adjoint matrix `Z11'` is the left inverse of the matrix `A11`.
"""
function enrichedaz(A11, A12, A21, A22, Z11star, b; options...)
    fact = enrichedazlinv_factorize(A11, A12, A21, A22, Z11star; options...)
    fact * b
end
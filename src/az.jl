
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
struct AZFactorization
    A                   # the A operator
    Zstar               # the Z' operator
    AminAZA             # A - AZ'A
    IminAZ              # I - AZ'
    R                   # set of random vectors
    az1_fact            # low-rank factorization of AminAZA*R
end

# apply AZ to a vector
function Base.:*(F::AZFactorization, b)
    # This is the AZ algorithm:
    # - step 1
    b1 = F.IminAZ*b
    u1 = F.az1_fact \ b1
    x1 = F.R * u1
    # - step 2
    b2 = b - F.A*x1
    x2 = F.Zstar * b2
    # - step 3
    x1+x2
end

"""
    az(A, Zstar, b)

Compute an AZ factorization with the given `(A,Z)` pair.

The function accepts the same options as the `az` function.
"""
function az_factorize(A, Zstar;
        tol             = az_tolerance(A),
        rank_estimate   = az_rank_estimate(A, Zstar; tol),
        adaptive_rank   = false)

    AminAZA = az_AminAZA(A, Zstar)
    IminAZ = az_IminAZ(A, Zstar)
    R, Col = randomized_column_space(AminAZA, rank_estimate)
    az1_fact = tsvd_factorize(Col, tol)
    AZFactorization(A, Zstar, AminAZA, IminAZ, R, az1_fact)
end

"""
    az(A, Zstar, b)

Apply the AZ algorithm to the right hand side vector `b` with the given
combination of `A` and `Z`. The matrix `Z` is given in its adjoint form `Z'`.
"""
function az(A, Zstar, b;
        tol             = az_tolerance(A),
        rank_estimate   = az_rank_estimate(A, Zstar; tol),
        adaptive_rank   = false)

    fact = az_factorize(A, Zstar; tol, rank_estimate, adaptive_rank)
    fact * b
end

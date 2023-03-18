
"A tolerance for use in the AZ algorithm."
az_tolerance(x) = az_tolerance(eltype(x))
az_tolerance(::Type{Complex{T}}) where T = az_tolerance(T)
az_tolerance(::Type{T}) where {T <: AbstractFloat} = 100eps(T)

# We have no idea what is a good default choice of the rank.
az_rank_estimate() = 20

"""
az_rank_estimate(A[, Zstar; tol])

Return an estimate of the rank of the system in step 1 of AZ, for the given
combination of `A` and `Z'`, and the given tolerance.
"""
az_rank_estimate(A, Zstar; tol = az_tolerance(A)) = az_rank_estimate(A; tol)
az_rank_estimate(A; tol = az_tolerance(A)) = az_rank_estimate()

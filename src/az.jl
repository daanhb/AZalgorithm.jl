
function tsvd(A, b; tol)
    u,s,v = svd(A; full=false)
    I = findlast(s/s[1] .>= tol)
    if I < size(A,2)
        u = u[:,1:I]
        v = ((v')[1:I,:])'
        s = s[1:I]
    end
    sinv = s.^(-1)
    v*(sinv .* (u'*b))
end

function randomized_column_space(A, r)
    m, n = size(A)
    T = eltype(A)
    Col = zeros(T, m, r)
    R = rand(real(T), n, r)
    for i in 1:r
        Col[:,i] = A*R[:,i]
    end
    R, Col
end

az_AminAZA(A::AbstractMatrix, Zstar::AbstractMatrix) = A - A*Zstar*LinearMaps.LinearMap(A)
az_IminAZ(A::AbstractMatrix, Zstar::AbstractMatrix) = I - A*LinearMaps.LinearMap(Zstar)

az_AminAZA(A, Zstar) = A - A*Zstar*A
az_IminAZ(A, Zstar) = I - A*Zstar


"""
    az(A, Zstar, b)

Apply the AZ algorithm to the right hand side vector `b` with the given
combination of `A` and `Z`. The matrix `Z` is given in its adjoint form `Z'`.
"""
function az(A, Zstar, b;
        tol             = az_tolerance(A),
        rank_estimate   = az_rank_estimate(A, Zstar; tol),
        adaptive_rank   = false,
        verbose         = false)

    m, n = size(A)
    AminAZA = az_AminAZA(A, Zstar)
    IminAZ = az_IminAZ(A, Zstar)
    az_core(A, Zstar, AminAZA, IminAZ, b; rank_estimate, tol)
end

function az_core(A, Zstar, AminAZA, IminAZ, b; rank_estimate, tol)
    R, Col = randomized_column_space(AminAZA, rank_estimate)
    u1 = tsvd(Col, IminAZ*b; tol)   # step 1
    x1 = R*u1
    x2 = Zstar*(b - A*x1)           # step 2
    x1+x2                           # step 3
end

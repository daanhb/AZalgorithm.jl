
"Approximate the column space of `A` using a set of random vectors."
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

 """
     tsvd(A, b; atol, rtol)

Compute the solution of `Ax=b` using total SVD with the given tolerance.

The total svd method discards all singular values small than the threshold.
"""
function tsvd(A, b, atol, rtol = atol)
    u,s,v = tsvd_factorize(A, atol, rtol)
    sinv = s.^(-1)
    x = v*(sinv .* (u'*b))
    x
end

"""
    u,s,v = tsvd_factorize(A, threshold)

Compute a tsvd factorization of the matrix `A`. This is like an SVD, but with
singular values (and associated singular vectors) below a threshold discarded.
"""
function tsvd_factorize(A, atol, rtol = atol)
    u,s,v = svd(A; full=false)
    threshold = max(atol, s[1]*rtol)
    I = findlast(s .>= threshold)
    if I < length(s)
        u = u[:,1:I]
        v = v[:,1:I]
        s = s[1:I]
    end
    SVD(u,s,v')
end

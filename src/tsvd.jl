
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
     tsvd(A, b, threshold)

Compute the solution of `Ax=b` using total SVD with the given tolerance.

The total svd method discards all singular values small than the threshold.
"""
function tsvd(A, b, threshold)
    u,s,v = tsvd_factorize(A, threshold)
    sinv = s.^(-1)
    x = v*(sinv .* (u'*b))
    x
end


"""
    u,s,v = tsvd_factorize(A, threshold)

Compute a tsvd factorization of the matrix `A`. This is like an SVD, but with
singular values (and associated singular vectors) below a threshold discarded.
"""
function tsvd_factorize(A, threshold)
    u,s,v = svd(A; full=false)
    I = findlast(s/s[1] .>= threshold)
    if I < length(s)
        u = u[:,1:I]
        v = v[:,1:I]
        s = s[1:I]
    end
    SVD(u,s,v')
end
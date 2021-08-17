function PR(A, d, α = 0.85, maxiters = 100, ϵ = 1.0e-4)
    n = size(A, 1)
    r = GBVector(n, 1.0 / n)
    t = GBVector{Float64}(n)
    d = copy(d)
    d[:, accum=/] = α
    teleport = (1 - α) / n
    for _ ∈ 1:maxiters
        temp = t; t = r; r = temp
        w = t ./ d
        r[:] = teleport
        mul!(r, A, w, (+, second), accum=+)
        eadd!(t, t, r, -)
        map!(abs, t)
        if reduce(+, t) <= ϵ
            break
        end
    end
    return r
end

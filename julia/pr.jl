function PR(A, d, α = 0.85, maxiters = 100, ϵ = 1.0e-4)
    n = size(A, 1)
    r = GBVector(n, 1.0 / n)
    t = GBVector{Float64}(n)
    d = d ./ α
    dmin = GBVector(n, 1.0 / n)
    eadd!(d, d, dmin, max)
    teleport = (1 - α) / n
    i = 0
    for j ∈ 1:maxiters
        temp = t; t = r; r = temp
        w = t ./ d
        r[:] = teleport
        mul!(r, A, w, (+, second), accum=+)
        eadd!(t, t, r, -)
        map!(abs, t)
        if reduce(+, t) <= ϵ
            break
        end
        i = j
    end
    print(i)
    return r
end

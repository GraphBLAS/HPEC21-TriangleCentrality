function TC1(A, d)
    T = mul(A, A', mask=A, desc=S)
    y = reduce(+, T, dims=2)
    k = reduce(+, y)
    return (3 .* mul(A, y) - 2 .* mul(one.(T), y) .+ y) ./ k
end

function TC3(A, d)
    M = tril(A, -1)
    T = mul(A, A', (+, pair), mask=M, desc=S)
    y = reduce(+, T, dims=2) .+ reduce(+, T', dims=2, desc=S)
    k = reduce(+, y)
    T2 = *(+, second)(T, y) .+ *(+, second)(T', y, desc=S)
    r = (3 .* *(+, second)(A, y)) + (-2 .* T2) + y
    return r ./ k
end

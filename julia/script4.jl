using SuiteSparseMatrixCollection
using MatrixMarket
using SuiteSparseGraphBLAS
SuiteSparseGraphBLAS.gbset(SuiteSparseGraphBLAS.FORMAT, SuiteSparseGraphBLAS.BYROW)
using BenchmarkTools
using SparseArrays
using LinearAlgebra
include("tc.jl")
include("pr.jl")
graphs = [
    #"karate",
    "com-Youtube",
    #"as-Skitter",
    #"com-LiveJournal",
    #"com-Orkut",
    "com-Friendster",
]

ssmc = ssmc_db()
matrices = filter(row -> row.name ∈ graphs, ssmc)
BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true
for name ∈ graphs
    path = fetch_ssmc(matrices[matrices.name .== name, :])[1]
    G = GBMatrix(convert(SparseMatrixCSC{Float64}, MatrixMarket.mmread(joinpath(path, "$name.mtx"))))
    SuiteSparseGraphBLAS.gbset(G, SuiteSparseGraphBLAS.FORMAT, SuiteSparseGraphBLAS.BYROW)
    GC.gc()
    G[:,:, mask=G, desc=SuiteSparseGraphBLAS.S] = 1
    diag(G)
    println("$name | $(size(G)) | $(nnz(G)) edges")
    d = reduce(+, G; dims=2)
    # for centrality in [PR, TC1, TC3]
    for centrality in [TC1]
        println("Benchmarking $(string(centrality)) on $(name)")
        i = 0.0
        j = @elapsed centrality(G, d) #warmup
        println("warmup $(j)")
        for run ∈ 1:3
            j = @elapsed centrality(G, d)
            println("trial time: $(j)")
            i += j
        end
        println("$(string(centrality)) on $(name) over 3 runs took an average of: $(i / 3)s")
    end
end

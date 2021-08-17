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
    "as-Skitter",
    "com-LiveJournal",
    "com-Orkut",
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
    for centrality in [PR, TC1, TC3]
        println("Benchmarking $(string(centrality)) on $(name)")
        result = @benchmark $centrality($G, $d) samples=3 seconds=70
        show(stdout,MIME"text/plain"(),result)
    end
end

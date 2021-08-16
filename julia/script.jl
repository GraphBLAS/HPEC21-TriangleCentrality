using SuiteSparseMatrixCollection
using MatrixMarket
using SuiteSparseGraphBLAS
SuiteSparseGraphBLAS.gbset(SuiteSparseGraphBLAS.FORMAT, SuiteSparseGraphBLAS.BYROW)
using BenchmarkTools
using SparseArrays
include("tc.jl")
include("pr.jl")
graphs = [
    "karate",
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
    G = GBMatrix(SparseArrays.sortSparseMatrixCSC!(convert(SparseMatrixCSC{Float64}, MatrixMarket.mmread(joinpath(path, "$name.mtx")))))
    SuiteSparseGraphBLAS.gbset(G, SuiteSparseGraphBLAS.FORMAT, SuiteSparseGraphBLAS.BYROW)
    show(stdout, MIME"text/plain"(), G)
    GC.gc()
    println("$name | $(size(G)) | $(nnz(G)) edges")
    for centrality in [PR, TC1, TC3]
        println("Benchmarking $(string(centrality)) on $(name)")
        result = @benchmark $centrality($G) samples=3 seconds=100
        show(stdout,MIME"text/plain"(),result)
    end
end

begin
    using Flux
    using Rotations
    using LinearAlgebra
    using RecursiveArrayTools
    using FiniteDiff
    #using math
    include("utils.jl")
end



begin
    v = Vector{Vector{Float64}}([zeros(3) for _ in 1:3]) # Vector{Vector{Float64}}(Vector{Float64}(undef, 3), 3)

    b = 39.0

    a = [1.0, 2.0, 3.0, b]
    c = a[2]



    a[2] = 90.0
    v[2] = a #[1:end]
    a[1] = 50
    #b = 100

    println(v)
end






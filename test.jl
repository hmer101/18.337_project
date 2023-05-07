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
    h = [1, 2, 3]
    a = [4, 5, 6]

    f = h./a

    println(f)
end






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
    A = [[1, 2], [3, 4, 5], [6], [7, 8, 9, 10], [11]]
    
    for i in eachindex(A[2])
        println(A[4][i])
    end

end






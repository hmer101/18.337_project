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
    # function f(x)
    #     return x^2 + sin(x)
    # end

    # x = 1.0
    # df = 0.0 #::Float64

    # grad = FiniteDiff.finite_difference_derivative(f,x) #finite_difference_gradient(f, x)

    # println("Gradient of f(x) at x=$x is $grad")

    #using Distributed, ArrayPartitions

    # Define the number of partitions and the length of each partition
    num_t_points = 2
    part_length = 100

    # Initialize the vector of ArrayPartitions
    ap = ArrayPartition([0.0, 0.0, 0.0], [1.1, 1.1, 1.1])

    #arr_parts = fill(ap, num_t_points) #fill(ArrayPartition{Float64}(part_length), num_t_points)
    arr_parts = [copy(ap) for _ in 1:num_t_points]
    println()
    println(arr_parts)

    arr_parts[1].x[1][1:length(arr_parts[1].x[1])] = [2.0, 2.0, 2.0]
    arr_parts[1] = ap #.x[1][1:length(arr_parts[1].x[1])] = [2.0, 2.0, 2.0]

    println()
    println(arr_parts)

    # Verify that each element is an ArrayPartition with the correct length
    # for i in 1:num_t_points
    #     @assert length(arr_parts[i]) == part_length
end






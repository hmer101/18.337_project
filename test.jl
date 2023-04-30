begin
    using Flux
    using Rotations
    using LinearAlgebra
    #using math
    include("utils.jl")
end

begin
    # function drone_forces_and_moments(num_drones::Int, t::Float64, f::Vector{Float64}, m::Matrix{Float64})
    #     # Preallocate the output vector
    #     v = Vector{Float64}(undef, num_drones + 3 * num_drones)
        
    #     # Store the force inputs for each drone
    #     for i in 1:num_drones
    #         v[i] = f[i]
    #         v[num_drones + 3*i - 2 : num_drones + 3*i] = m[:, i]
    #     end
    
    #     return v
    # end
    
    # # Test the function with sample data
    # # num_drones = 3
    # # t = 1.0
    # # f = [1.0, 2.0, 3.0] # Vector with force inputs for each drone
    # # m = [4.0 7.0 10.0; 5.0 8.0 11.0; 6.0 9.0 12.0] # 3x3 matrix with moment inputs for each drone
    
    # # v = drone_forces_and_moments(num_drones, t, f, m)
    # # sum_R_L_T_i_load =  zeros(3)
    # # sum_R_L_T_i_load -= [4, 4, 4]
    # # u0 = [rand(2, 2) for _ in 1:3]


    # # nn_model1 = Chain(Dense(12, 32, tanh), Dense(32, 8))

    # # # Example initial conditions and time span (assuming u contains 2x2 matrices)
    # # u0 = [rand(2, 2) for _ in 1:3]
    # # tspan = (0.0, 1.0)

    # # #out = nn_model1(vcat(u0))


    # # a = 1.2 
    # # b = 0.2
    # # c = -0.4

    # # euler_angles = EulerAngles(a, b, c)

    # # # Convert Euler angles to a rotation matrix
    # # rot_matrix = Rotation(euler_angles)
    # # println(rot_matrix)

    # R = RotZYX(a, b, c) #RotZYX(a, b, c) #RotZYX(a, b, c)
    # println(R)

    # # pitch=atan(-R[3,1],sqrt(R[3,2]^2+R[3,3]^2))
    # # pitch_gpt = asin(-R[3,1])

    # yaw=atan(R[2,1],R[1,1])
    # pitch=atan(-R[3,1],sqrt(R[3,2]^2+R[3,3]^2))
    # roll=atan(R[3,2],R[3,3])

    # # println(yaw)
    # # println(pitch)
    # # println(roll)
    # # println(Rotations.params(RotZYX(R)))



    # #print(R.y)


    # vector_a = [1, 2, 3]
    # vector_b = [4, 5, 6]
    # #v = ones(4)
    # print(cross(vector_a,vector_b))
    # print(hat_map(vector_a)*vector_b)

    # I3 = Matrix{Float64}(I, 3, 3)
    # t = [1 2 3]
    # println(zeros(3))
    # println(sqrt(14))

    #M = [Matrix{Float64}(I, 3, 3) for _ in 1:2, _ in 1:3]
    # M = [zeros(3) for _ in 1:2]
    # println("next")
    # println(M)



    # function flatten_M(M)
    #     nrows, ncols = size(M) # Get the dimensions of the input matrix
    
    #     # Calculate the total size of the output matrix
    #     total_rows = nrows * size(M[1, 1])[1]
    #     total_cols = ncols * size(M[1, 1])[2]
    
    #     # Initialize M_flat with the correct type and size
    #     M_flat = Matrix{eltype(M[1, 1])}(undef, total_rows, total_cols)
    
    #     # Iterate over the input matrix elements (matrices)
    #     for i in 1:nrows
    #         for j in 1:ncols
    #             row_range = ((i - 1) * size(M[1, 1])[1] + 1):i * size(M[1, 1])[1]
    #             col_range = ((j - 1) * size(M[1, 1])[2] + 1):j * size(M[1, 1])[2]
    #             M_flat[row_range, col_range] = M[i, j]
    #         end
    #     end
    
    #     return M_flat
    # end


    # function unflatten_M(M_flat, nrows, ncols, row_size, col_size)
    #     # Initialize the output matrix of matrices with the correct type and dimensions
    #     M = Matrix{Matrix{eltype(M_flat)}}(undef, nrows, ncols)
    
    #     # Iterate over the output matrix elements (matrices)
    #     for i in 1:nrows
    #         for j in 1:ncols
    #             row_range = ((i - 1) * row_size + 1):i * row_size
    #             col_range = ((j - 1) * col_size + 1):j * col_size
    #             M[i, j] = M_flat[row_range, col_range]
    #         end
    #     end
    
    #     return M
    # end

    # # Define a matrix of matrices
    # M = [reshape(collect(1:9), (3, 3)) for _ in 1:2, _ in 1:2]

    # # Flatten the matrix of matrices
    # M_flat = flatten_M(M)

    # # Unflatten the flattened matrix to recover the original matrix of matrices
    # M_recovered = unflatten_M(M_flat, 2, 2, 3, 3)

    # println("Original matrix of matrices:")
    # println(M)
    # println("\nFlattened matrix:")
    # println(M_flat)
    # println("\nRecovered matrix of matrices:")
    # println(M_recovered)

    # push!(v,5.0)

    # #cross_product = cross(vector_a, vector_b)

    # #println("Cross product: ", cross_product)
    # println(v)
    #println("Star product: ", vector_a*vector_b)

    # Define the flattened vector of vectors
    v_flat = collect(1.0:9.0)

    # Define the desired shapes as a vector of tuples
    shapes = [(3,) for _ in 1:3]

    # Call the unflatten_v function with the specified inputs
    #unflattened_vectors = unflatten_v(v_flat, shapes)
    unflattened_vectors = unflatten_to_vec_of_vecs(v_flat, 3)

    # Print the unflattened_vectors
    println("Unflattened vectors:")
    println(unflattened_vectors)
  

    # mutable struct MyStruct
    #     a::Int
    #     b::Float64
    #     c::String
    
    #     function MyStruct(; a::Int=0, b::Float64=0.0, c::String="")
    #         return new(a, b, c)
    #     end
    # end

    #p = Person_init(name="Alice", age=32, height=1.7)
    # p.name = "hi"
    # p.age = 39
    # p.height = 1.8
    #m = [2.32 4.0 0; 0 2.32 0; 0 0 4]
    # r_cables = vcat([-0.42, -0.27, 0], [0.48, -0.27, 0], [-0.06, 0.55, 0])
    # r_cables = convert.(Float32, r_cables)

    # print(r_cables)
    
    
    # t = [Vector{Float64}(undef, 3) for i in 1:4] #Vector{Float64}(undef, 10)
    # t[2] = [1.0, 2.0, 3.2]
    # print(t)

end

# begin
#     mutable struct Person
#         name::String
#         age::Int
#         height::Float64
#     end

#     function Person_init(; name::String, age::Int=0, height::Float64=1.5)
#         return Person(name, age, height)
#     end
# end
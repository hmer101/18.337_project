# Utility functions file
# 
# Author: Harvey Merton

#####
# Vector and matrix conversion helper functions
#####

## Flattening and unflattening matricies and vectors
# Flattens vector of vectors
function flatten_v(v)
    n = 0 # v_flat index

    # Calculate the total length of the output vector
    total_len = sum(length(x) for x in v)
    
    # Initialize v_flat with the correct type and size
    v_flat = Vector{eltype(v[1])}(undef, total_len)
    
    for i in 1:length(v) # Iterate over the arrays
        l = length(v[i])
        v_flat[n+1:n+l] = reshape(v[i], (l,)) # Corrected index
        n += l
    end

    return v_flat
end

# Flatten matrix of matricies or matrix of vectors
function flatten_m(M)
    nrows, ncols = size(M) # Get the dimensions of the input matrix

    # Determine if the input contains matrices or vectors
    is_matrix = eltype(M) <: AbstractMatrix

    if is_matrix
        total_rows = nrows * size(M[1, 1])[1]
        total_cols = ncols * size(M[1, 1])[2]
    else  # Input contains vectors
        total_rows = nrows * size(M[1, 1])[1]
        total_cols = ncols
    end

    # Initialize M_flat with the correct type and size
    M_flat = Matrix{eltype(M[1, 1])}(undef, total_rows, total_cols)

    # Iterate over the input matrix elements (matrices or vectors)
    for i in 1:nrows
        for j in 1:ncols
            if is_matrix
                row_range = ((i - 1) * size(M[1, 1])[1] + 1):i * size(M[1, 1])[1]
                col_range = ((j - 1) * size(M[1, 1])[2] + 1):j * size(M[1, 1])[2]
                M_flat[row_range, col_range] = M[i, j]
            else  # Input contains vectors
                row_range = ((i - 1) * size(M[1, 1])[1] + 1):i * size(M[1, 1])[1]
                M_flat[row_range, j] = M[i, j]
            end
        end
    end

    return M_flat
end



# Unflatten vector of matricies or vector of vectors
# function unflatten_v(v_flat::Vector{T}, shapes::Vector{Tuple{Int, Vararg{Int64}}}) where T
#     v = Vector{Any}(undef, length(shapes)) # Initialize output vector #Matrix{T}
#     n = 1 # v_flat index

#     for i in 1:length(shapes)
#         shape = shapes[i]

#         # Slice flattened array to get next section
#         # Handle case for vector rather than matrix size
#         l = shape[1]
#         if length(shape) == 2 # Matrix
#             l = l * shape[2]
#             # Reshape to make into a matrix
#             v[i] = reshape(v_flat[n:n+l-1], shape) # Reshape the slice of v_flat to create the matrix
#         else # Vector
#             # No reshaping required - just assign
#             v[i] = v_flat[n:n+l-1]
#         end
   
#         n += l
#     end

#     return v
# end

# Simplified version of above function: unflatten vector of vectors
function unflatten_v(v_flat::Vector{Float64}, num_elements::Int)
    num_vectors = div(length(v_flat), num_elements)
    v = [v_flat[num_elements*i-(num_elements-1):num_elements*i] for i in 1:num_vectors]
    return v
end


# Unflatten matrix of matricies or matrix of vectors
function unflatten_m(M_flat, nrows, ncols, row_size, col_size, is_matrix::Bool)
    # Initialize the output matrix of matrices with the correct type and dimensions
    M = Matrix{Vector{eltype(M_flat)}}(undef, nrows, ncols)

    # Iterate over the output matrix elements (matrices or vectors)
    for i in 1:nrows
        for j in 1:ncols
            if is_matrix
                row_range = ((i - 1) * row_size + 1):i * row_size
                col_range = ((j - 1) * col_size + 1):j * col_size
                M[i, j] = M_flat[row_range, col_range]
            else  # Output contains vectors
                row_range = ((i - 1) * row_size + 1):i * row_size
                M[i, j] = M_flat[row_range, j]
            end
        end
    end

    return M
end

#####
## Kinematics and vector geometry helper functions
#####

# Checks if two vectors are the same but pointing in opposite directions
function are_opposite_directions(v1::Vector, v2::Vector)
    # Normalize both vectors
    norm_v1 = v1 ./ norm(v1)
    norm_v2 = v2 ./ norm(v2)

    # Check if the normalized vectors are negatives of each other
    return isapprox(norm_v1, -norm_v2)
end

# Calculates the hat map of a vector
function hat_map(v::Vector)
    if length(v) != 3
        error("The input vector must have exactly 3 elements.")
    end

    V = [  0   -v[3]  v[2];
          v[3]   0   -v[1];
         -v[2]  v[1]   0  ]
    return V
end


# Convert RPY Euler angles to a rotation matrix R
function rpy_to_R(rpy::Vector{T}) where T
    # Check that the input vector is of size 3 (i.e. contains RPY)
    if length(rpy) != 3
        throw(ArgumentError("The input vector must have exactly 3 elements"))
    end
    
    # Convert the RPY Euler angles to a rotation matrix R
    return RotZYX(rpy[1], rpy[2], rpy[3])
    
end

# Convert a rotation matrix R to RPY Euler angles
function rpy_to_R(R::Matrix{T}) where T
    # Check that the input matrix is of size 3x3. Could do further checks to see if is special orthogonal
    if size(R) != (3, 3)
        throw(ArgumentError("The input matrix must be of size 3x3"))
    end
    
    # Convert the RPY Euler angles to a rotation matrix R
    return Rotations.params(RotZYX(R))
    
end


#####
# Results visualization
#####

# Plot data from a vector containing num_drones x 3-vectors over a time span given in t_data 
function plot_data!(plot_var, t_data, data::Vector{Vector{Vector{Float64}}}, plot_components::Bool, x_domain::Tuple{Float64, Float64}, label_prefix::String, y_label::String)
    # Loop through all drones/cables
    for drone_ind in 1:length(data[1])
        # Extract x,y,z components
        comp1 = [(@view data[i][drone_ind][1])[] for i in 1:length(data)]
        comp2 = [(@view data[i][drone_ind][2])[] for i in 1:length(data)]
        comp3 = [(@view data[i][drone_ind][3])[] for i in 1:length(data)]

        # Find the magnitude
        magnitude = [sqrt(comp1[i]^2 + comp2[i]^2 + comp3[i]^2) for i in 1:length(data)]

        # Add to plot
        two_dp_formatter(x) = @sprintf("%.2f", x)

        if plot_components
            # Plot components for all time points
            plot!(plot_var, t_data, [comp1, comp2, comp3], xlims=x_domain, seriestype = :scatter, legend = :topleft, label = [string("$label_prefix$drone_ind", "_x") string("$label_prefix$drone_ind", "_y") string("$label_prefix$drone_ind", "_z")], xlabel = "Time (s)", ylabel = "$y_label", yformatter=two_dp_formatter)
        else
            # Plot magnitude for all time points
            plot!(plot_var, t_data, magnitude, xlims=x_domain, seriestype = :scatter, legend = :topleft, label = "$label_prefix$drone_ind", xlabel = "Time (s)", ylabel = "$y_label", yformatter=two_dp_formatter) #right_margin = 5mm) #margin=(0mm, 5mm, 0mm, 0mm))
        end

    end

    return plot_var

end

# Plot data generated for training the NN
function plot_results(t_data, T_data::Vector{Vector{Vector{Float64}}}, x₍i_rel_Lᵢ₎::Vector{Vector{Vector{Float64}}}, ẋ₍i_rel_Lᵢ₎::Vector{Vector{Vector{Float64}}}, ẍ₍i_rel_Lᵢ₎::Vector{Vector{Vector{Float64}}}, plot_components::Bool)
    x_domain = (t_data[1], t_data[end]+((t_data[2] - t_data[1])/10))
    
    ## Plot tension data
    p_tension = plot()
    plot_data!(p_tension, t_data, T_data, plot_components, x_domain, "T_data_","Tension (N)")
    
    ## Plot trajectory data
    # Position
    p_x = plot()
    plot_data!(p_x, t_data, x₍i_rel_Lᵢ₎, plot_components, x_domain, "x₍i_rel_Lᵢ₎_", "Location (m)")

    # Velocity - first value will be off (backwards FD) so don't include
    p_ẋ = plot()
    plot_data!(p_ẋ, t_data[2:end], ẋ₍i_rel_Lᵢ₎[2:end], plot_components, x_domain, "ẋ₍i_rel_Lᵢ₎_", "Velocity (m/s)")

    # Acceleration - first two values will be off (backwards FD) so don't include
    p_ẍ = plot()
    plot_data!(p_ẍ, t_data[3:end], ẍ₍i_rel_Lᵢ₎[3:end], plot_components, x_domain, "ẍ₍i_rel_Lᵢ₎_", "Acceleration (m/s²)")


    ## Display plots
    # Tensions
    display(p_tension)

    # Trajectory
    p_traj = plot(p_x, p_ẋ, p_ẍ, layout=(3,1), size=(800, 600))
    display(p_traj)

end



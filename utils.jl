# Utility functions file
# 
# Author: Harvey Merton


# Flattens vector
function flatten(v)
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

# Unflatten vector
function unflatten(v_flat::Vector{T}, shapes::Vector{Tuple{Int, Vararg{Int64}}}) where T
    v = Vector{Any}(undef, length(shapes)) # Initialize output vector #Matrix{T}
    n = 1 # v_flat index

    for i in 1:length(shapes)
        shape = shapes[i]

        # Slice flattened array to get next section
        # Handle case for vector rather than matrix size
        l = shape[1]
        if length(shape) == 2 # Matrix
            l = l * shape[2]
            # Reshape to make into a matrix
            v[i] = reshape(v_flat[n:n+l-1], shape) # Reshape the slice of v_flat to create the matrix
        else # Vector
            # No reshaping required - just assign
            v[i] = v_flat[n:n+l-1]
        end
   
        n += l
    end

    return v
end
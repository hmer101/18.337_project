# A few different numerical methods that I needed to handle the data

using DSP # For smoothing (convolution)


# Smooth data using convolution and windowing TODO This is straight ripped from the internet
function smooth(y :: Array{T, 1} where T; win_len = 71, win_method = 2) # http://www.simulkade.com/posts/2015-05-07-how-to-smoothen-noisy-data.html

    # Sanitize the window to be of same type as the elements of y
    dtype = eltype(y)
    #println("Data type is " * string(dtype))

    # Check input makes sense
    if win_len%2 == 0
        win_len += 1 # only use odd numbers
    end

    # Generate the window
    # win_method : 1: flat, 2: hanning, 3: hamming ...
    if win_method == 1
        w = ones(dtype, win_len)
    elseif win_method == 2
        w = dtype.(DSP.hanning(win_len))
    elseif win_method == 3
        w = dtype.(DSP.hamming(win_len))
    end

    if win_len < 3
        println("Bad window length (win_len < 3)")
        return y
    elseif length(y) < win_len
        println("Bad window length (length(y) < win_len)")
        return y
    else
        y_new = [2*y[1] .- reverse(y[1:win_len],1); y[:]; 2*y[end] .- reverse(y[end-win_len:end],1)]
        y_smooth = DSP.conv(y_new, w/sum(w))
        ind = floor(Int, 1.5*win_len)
        #println(typeof(y_smooth))
        #println(eltype(y_smooth))
        return y_smooth[1+ind:end-ind-1]
    end

end


# Smooth data that is provided as an Array{T, 2}
function smooth(d :: Array{T, 2} where T; win_len = 71, win_method = 2, visuals = true)

    # Perform smoothing on each row of d
    d_smoothed = similar(d)
    for i in 1:size(d, 1)
        d_smoothed[i, :] .= smooth(d[i, :], win_len = win_len, win_method = win_method)
    end

    # Plot output if desired
    if visuals # TODO I think this can be a single plot command
        p = plot(d', title = "Smoothed Data")
        plot!(d_smoothed', xlabel = "Time")
        display(p)
    end

    return d_smoothed
end


# Use central differencing to find the first-order numerical derivative of evenly-spaced data
# Kinda mad I had to implement this by hand! Apparently Julia doesn't have something like this...
function finite_difference(data :: Array{T, 2} where T, dt; num_pts = 5, repeat_ends = true)
    # repeat_ends...
        # If false: The dataset will be shrunk by (num_pts - 1), with (num_pts - 1)/2 removed from the beginning and end
        # If true: The dataset will have duplicated derivatives filling in dummy values at the start and end of the array
    # data should have each column represent a single instant, with derivatives taken between columns
    # dt is a scalar, the time difference between each data point
    # num_pts can be 3, 5, 7 or 9, corresponding to the number of data points used to compute a single derivative
    # Reference for finite differencing: https://en.wikipedia.org/wiki/Numerical_differentiation and https://en.wikipedia.org/wiki/Finite_difference_coefficient

    # Hack to quickly get the indices needed
    inds = [0, 0, 1, 0, 2, 0, 3, 0, 4]

    # Sanitize constants to have proper type
    dtype = eltype(data)
    #println("Data type is " * string(dtype))

    # Coefficients for different central differencing schemes
    stencils = [
        dtype.([-0.5, 0.0, 0.5]),
        dtype.([1/12, -2/3, 0, 2/3, -1/12]),
        dtype.([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60]),
        dtype.([1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280])
    ]

    # Get the shifts set up correctly to either buffer or not buffer the ends
    split = Int64((num_pts-1)/2)
    if repeat_ends
        derivative = similar(data)
        offset = split
        shift = 0
    else
        derivative = Array{dtype, 2}(undef, (size(data, 1), size(data, 2) - num_pts + 1) )
        offset = 0
        shift = split
    end

    # Calculate the derivative
    l = size(derivative, 2)
    for i in (1 + offset):(l - offset)
        #@show data[:, (i-split+shift):(i+split+shift) ]
        derivative[:, i] .= @view(data[:, (i-split+shift):(i+split+shift) ]) * stencils[inds[num_pts]] ./ dt
        # This is matrix multiplication: (weight * column + weight * column + ...) / dt
    end

    # Fill in buffer entries if requested
    if repeat_ends
        for i in 1:offset
            derivative[:, i] .= derivative[:, 1 + offset]
        end
        for i in (l-offset+1):l
            derivative[:, i] .= derivative[:, l - offset]
        end
    end

    return derivative

end

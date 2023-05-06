using Dates
using LinearAlgebra
using Statistics
using Plots

# Datatype to hold information relevant to impedance measurements
struct Impedance{T}
    name :: String
    times :: Vector
    R :: Array{T, 2} # Resistance
    X :: Array{T, 2} # Reactance
    Z :: Array{T, 2} # Impedance magnitude
    ϕ :: Array{T, 2} # Impedance phase
    f :: Array{T, 1} # Frequencies
end


## Calculating impedance

# Calculate the impedance given data files, and return a unified data type
function calculate_impedance(f :: Array{Float64, 1}, R_file :: DataFile, X_file :: DataFile, fname_range :: UnitRange, visuals :: Bool)
    # R_file, X_file : Data files as prepared in read_files.jl, corresponding to resistance and reactance measurements
    # fname_range : Indices in each file name (from R_file) from which to draw a permanent name
    # The data is assumed to initially be time-advancing along dimension 1, and to need to be transposed to be time-advancing along dimension 2

    # Get name
    name = R_file.fname[fname_range]

    # Get times
    df = DateFormat("y-m-d_H:M:S")
    times = DateTime.(R_file.row_labels[2:end], df)

    # Get R, X by mutably transposing (to avoid lazy Transpose data type)
    # Also trim out the 101st data bin (which is garbage) # TODO make this a parameter / argument
    # Also trim out the first 30 data points (rapid temperature shift) # TODO make this a parameter / argument
    # R and X must be the same size and data type
    T = typeof(R_file.data[1, 1]) # Typically Float64
    dims = reverse(size(R_file.data)) # For transpose
    R = Array{T}(undef, dims)#(dims[1]-1, dims[2]-30))
    LinearAlgebra.transpose!(R, R_file.data)#R_file.data[31:end, 1:100])
    X = Array{T}(undef, dims)#(dims[1]-1, dims[2]-30))
    LinearAlgebra.transpose!(X, X_file.data)#X_file.data[31:end, 1:100])

    # Calculate Z, ϕ
    Z = (R.^2 .+ X.^2).^0.5
    ϕ = atan.(X ./ R)

    if visuals
        # It's cleaner to allow the user to call visualize_impedance() themselves if they want to change the optional settings
        visualize_impedance(name, times, Z, ϕ, f)
    end

    return Impedance(name, times, R, X, Z, ϕ, f)

end

# Calculate multiple impedances in one call (multiple overload)
# This could be broadcasted, but I'm going to keep it as is because of additional utility
function calculate_impedance(f, data :: Vector{Vector{DataFile}}; fname_range = 13:16, visuals = false)

    impedances = Vector{Impedance}(undef, length(data))
    for i in 1:length(data)
        impedances[i] = calculate_impedance(f, data[i][1], data[i][2], fname_range, visuals)
    end

    return impedances

end

# Plot raw gain and phase data
    # Overlaid line plots of spectra over time
    # Spectrogram heatmaps
function visualize_impedance(name, times, Z, ϕ, f; skip_by = 20, fancy_figure = true)

    # Set the indices (in time) to choose for the line plots
    s = size(Z[1:end, 1:skip_by:end])

    # Set the layout for the plots
    l = @layout [ a b ; c ; d ]

    if fancy_figure == false # Original view (includes dates & times of day)

        # Display settings
        plot_size = (1920, 1920)
        dpi = 300

        # Heatmap settings
        xlabels = times
        ylabels = []

        # Line plots
        x_values = 1:s[1]
        pZ_line = plot(x_values, Z[1:end, 1:skip_by:end], legend = false, title = "Impedance Z (Ohm)", yaxis = :log10, color = :roma, line_z = (1:s[2])')
        pϕ_line = plot(x_values, ϕ[1:end, 1:skip_by:end] .* 180 / (2π), legend = false, title = "Phase ϕ (deg.)", color = :roma, line_z = (1:s[2])')

    else # Publication-quality

        # Display settings
        plot_size = (1000, 1200)
        dpi = 600

        # Heatmap settings
        xlabels = round.(convert(Array{Float64}, Dates.value.(times - times[1]) ) / (1000.0 * 60 * 60); digits = 2) # Recall Julia is 1-based indexing
        ylabels = f
        
        # Line plots
        x_values = f
        pZ_line = plot(x_values, Z[1:end, 1:skip_by:end], legend = false, yaxis = :log10, color = :roma, line_z = (1:s[2])', xaxis = :log10, xlabel = "Frequencies (Hz)", ylabel = "Gain (Ω)", framestyle = :box)
        pϕ_line = plot(x_values, ϕ[1:end, 1:skip_by:end] .* 180 / (2π), legend = false, color = :roma, line_z = (1:s[2])', xaxis = :log10, xlabel = "Frequencies (Hz)", ylabel = "Phase (°)", framestyle = :box)

    end

    # Create heatmaps
    pZ = plot_heatmap(data_relative(Z), xlabels = xlabels, ylabels = ylabels, symmetric = true, cmap = :RdBu, title = "Gain: log₁₀(|Z| / |Z₀|)")
    pϕ = plot_heatmap(data_relative(ϕ .* 180 / (2π), log_first = false), xlabels = xlabels, ylabels = ylabels, symmetric = true, cmap = :RdBu, title = "Phase: ϕ - ϕ₀ (°)")

    # Display full plot with layout
    main_plot = plot(pZ_line, pϕ_line, pZ[1], pϕ[1], layout = l, size=plot_size, dpi = dpi)
    display(main_plot)

    # Save vector graphic (if fancy)
    if fancy_figure
        #savefig(main_plot, "impedance_raw_" * name * ".pdf")
        savefig(main_plot, "impedance_raw_" * name * ".png")
    end

end

## Verifying time-spacing of impedance measurements

# Make sure the timestamps are roughly uniformly spaced
# Return the standard deviation of measurement time periods, as fraction of average measurement time
function check_time_spacing(data :: Impedance)
    times = data.times
    dt = (t->t.value).(times[2:end] - times[1:end-1])
    dt_std = Statistics.std(dt)
    return dt_std / Statistics.mean(dt)
end

# Get the time, in milliseconds, since the start of the experiment
function get_relative_times(data :: Impedance)
    return (t->t.value).(data.times[1:end] .- data.times[1])
end

# TODO this was another unnecessary one
# A wrapper for the above that checks multiple time spacings at once
#function check_time_spacing(data :: Vector{Impedance})
#    out = Vector{Float64}(undef, length(data))
#    for i in 1:length(data)
#        out[i] = check_time_spacing(data[i])
#    end
#    return out
#end

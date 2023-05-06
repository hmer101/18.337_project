using Plots
using Dates

# Plots a heatmap using standardized formatting on a specified axis
# TODO: Save all the heatplots automatically, to a common folder where they can be easily retrieved if desired
function plot_heatmap(data :: Array; xlabels :: Vector = [], ylabels :: Vector = [], symmetric :: Bool = false, scale :: AbstractFloat = NaN, scale_clip :: AbstractFloat = 0.5, cmap = :BrBG, title :: String = "Heatmap")
    # data : 2D array of numeric data
    # xlabels : either [], a list of strings, or a list of TimeType. If you specifically use DateTime, it tries a little harder to format your times nicely.
    # ylabels : either [] or a list of strings
    # If either label are empty, the labels will be indices
    # symmetric : should the color range be symmetric about 0?
    # scale : IF symmetric, you can specify a magnitude for the maximum and minimum values
    # scale_clip : IF symmetric, and regardless of whether scale is specified, you can specify a clipping level (between 0 and 1) to cut out extremes and allow you to see middle tones

    # Sanitize the xlabels to be lists of strings
    xrotation = 0
    if xlabels == []
        # Create numbers
        xlabels = (x->string(x)).(1:size(data, 2))
        xaxis_label = ""
    elseif typeof(xlabels[1]) == DateTime # Specifically DateTime   #if typeof(xlabels[1]) <: TimeType # Any type of time
        # Create nice, intelligible date & time labels
        xlabels = (t->string(Dates.format(t, "yyyy-mm-dd HH:MM:SS"))).(xlabels)
        xrotation = 45
        xaxis_label = ""
    else
        # Create strings, at least
        #xlabels = (x->string(x)).(xlabels)
            # This is actually not desirable; forces it to grab really weird x labels
            # Better to leave it as numbers; then it grabs nice round numbers
        xaxis_label = "Time (hours)"
    end

    # Sanitize the ylabels to be lists of strings
    if ylabels == []
        # Create numbers
        ylabels = (y->string(y)).(1:size(data, 1))
        yticks = []

    else

        # ASSUME if numbers are provided for the y axis that this is a spectrogram, and we have log-spaced frequencies
        # Create custom exponential labels for the frequency axis
            # Using the :log10 spacing does not work (glitched) - distorts the heatmap, but the labels are correct
            # Using numbers for the labels on a linear scale distorts the heatmap in a different way
            # Could use LaTeX, but that breaks sometimes
                # Can't use it with GR backend, and I'm using GR for everything
            # The following is very much a workaround; uses pure unicode characters to achieve exponentials (10^x)

        ylabels_old = deepcopy(ylabels) # The actual frequency values
        ylabels = 1:length(ylabels) # Just a list of indices

        # Find the range of orders of magnitude in frequency
        low = ceil(log10(minimum(ylabels_old)))
        high = floor(log10(maximum(ylabels_old)))
        exponents = low:1:high
        
        # Generate some warnings if appropriate
        if low == high || low > high
            println("WARNING: Auto-ranging exponential scaling will only work if you have at least a few orders of magnitude range in frequency")
        end
        if low < -9 || high > 9
            println("WARNING WARNING WARNING: IF YOU HAVE FREQUENCIES ABOVE 10^9 OR BELOW 10^-9, THE HEATMAP AXIS LABEL HACK WON'T WORK")
            println("        It can be fixed: instead of just looking at the exponent as a single character, look at it as a string, and use the weird unicode superscript characters for each character of the string. Not going to do it now because it's a pain.")
        end

        # Kind of brute force: Find the data indices that correspond to integer power-of-ten frequencies
        ytickvalues = zeros(length(exponents))
        for i in 1:length(exponents)
            for j in 1:length(ylabels_old)
                if round(ylabels_old[j]) == 10^exponents[i]
                    ytickvalues[i] = j
                end
            end
        end
        yticks = (ytickvalues, (exponent->"10"*superscriptnumber(round(Int, exponent))).(exponents))
   end

    # Set the maximum and minimum color levels
    if symmetric
        if isnan(scale)
            scale = max(abs(minimum(data)), abs(maximum(data)))
        end
        vmin = - scale_clip * scale
        vmax =   scale_clip * scale
    else
        # This is kind of a do-nothing to recreate default behavior
        vmin = minimum(data)
        vmax = maximum(data)
    end

    # Actually plot something
    hhh = heatmap(xlabels, ylabels, data, c = cmap, xlabel = xaxis_label, xrotation = xrotation, clims = (vmin, vmax), colorbar_title = title, framestyle = :box, ticks = :native)#, minorticks = true, ticks = true)
    # Set the custom yticks value, if it's been created; otherwise, let the default ticks stay
    if yticks != []
        yticks!(yticks)
        ylabel!("Frequency (Hz)")
    end

    # Return the plot object (don't display it right away), plus some other parameters
    return hhh, scale, (vmin, vmax)

end


# Workaround to get supercripts; From https://stackoverflow.com/questions/69485556/how-to-modify-superscripts-in-julia-labels-of-gadfly-plots
# Gets you a positive or negative one-digit integer superscript
function superscriptnumber(i::Int)
    if i < 0
        c = [Char(0x207B)]
    else
        c = []
    end
    for d in reverse(digits(abs(i)))
        if d == 0 push!(c, Char(0x2070)) end
        if d == 1 push!(c, Char(0x00B9)) end
        if d == 2 push!(c, Char(0x00B2)) end
        if d == 3 push!(c, Char(0x00B3)) end
        if d > 3 push!(c, Char(0x2070+d)) end
    end
    return join(c)
end

function subscriptnumber(i::Int)
    if i < 0
        c = [Char(0x207B)]
    else
        c = []
    end
    for d in reverse(digits(abs(i)))
        if d == 0 push!(c, Char(0x2080)) end
        #if d == 1 push!(c, Char(0x2081)) end
        #if d == 2 push!(c, Char(0x2082)) end
        #if d == 3 push!(c, Char(0x2083)) end
        if d > 0 push!(c, Char(0x2080+d)) end
    end
    return join(c)
end




# Perform "exposure correction" to get heatplot-ready data that clearly illustrates the differences between two sets
function data_relative(data_in :: Array; log_first :: Bool = true, log_after :: Bool = false, average_at :: Int = 20, average_until :: Int = 30)#average_at :: Int = 600, average_until :: Int = 900)#

    # Turn the data to logarithmic scale (or not)
    data = similar(data_in)
    if log_first
        data .= log10.(data_in)
    else
        data .= data_in
    end

    # Subtract out the average over an early region where the data has settled
    avg = sum(data[1:end, average_at:average_until], dims = 2) / (average_until - average_at + 1) # The +1 is because the endpoint is included in Julia's indexing system
    data .= data .- avg # .- of vector and matrix subtracts the vector from all columns of the matrix

    # Turn the data to logarithmic scale (or not)
    if log_after
        data .= log10.(data)
    end

    # The data is now relative to an average, and is on logarithmic scale if desired
    return data

end

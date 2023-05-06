using Statistics
using MultivariateStats


# Store the PCA results that will be useful going forward
# (We won't need the model again during training, but after training, we'll need it to transform incoming data)
struct PCA_results{T}
    model :: PCA # This can be used to transform other data, and to grab e.g. projection matrices
    transformed :: Array{T, 2} # This is the simplified representation of the input data, which will be used in learning TODO, would be cool to give this the option of being a vector of these boys. Maybe use multiple dispatch to make it happen.
end

struct PCA_results_multi{T} # An overload that can hold multiple datasets for a single model
    model :: PCA
    transformed :: Vector{Array{T, 2}}
end


function PCA_plot(f, times, data, model, transformed, log_first, log_after; restore_mean = false, fancy_plot = false, name = "unnamed", data_ylabel = "", is_gain = true, dataset_mean_vector = [], save_fancy_figures = false)

    ### Perform some calculations to show PCA reconstructions

    # Reconstruct an approximation for the original data
    reconstructed = reconstruct(model, transformed)
    if dataset_mean_vector == []
        dataset_mean_vector = sum(data, dims = 2) / size(data, 2) # We don't store the average because it would have required another allocation
    end
    if restore_mean # This means the model does NOT contain the dataset's original mean, so we have to reintroduce it
        reconstructed .+= dataset_mean_vector
    end

    # Reconstruct them sequentially from nothing 
    if fancy_plot
        
        num_princ_components = size(transformed, 1)
        
        # Reconstruct using variable numbers of the available principal components
        reconstruction_labels = Array{String, 1}(undef, num_princ_components + 2)
        reconstructions = Array{Array{Float64, 2}, 1}(undef, num_princ_components + 2)
        reconstructions[1] = data
        reconstruction_labels[1] = "Original"
        for i in 1:(num_princ_components+1)
            transformed_i = deepcopy(transformed)
            transformed_i[i:end, :] .= 0 # Kill the principal components of index i or greater
            reconstructions[i+1] = reconstruct(model, transformed_i)
            reconstructions[i+1] .+= dataset_mean_vector
            if i == 1
                reconstruction_labels[i+1] = "Mean"
            else
                reconstruction_labels[i+1] = string(i - 1)
            end
        end

        # Select which timestamp (spectrum index) you would like to use to demonstrate
        reconstruction_slices = Array{Array{Float64, 1}, 1}(undef, num_princ_components + 2)
        for i in 1:(num_princ_components + 2)
            reconstruction_slices[i] = reconstructions[i][:, end]
            # Use the final spectrum, as it should have the greatest difference from the initial spectrum
        end

    end


    # Calculate the deviation (residuals / original)
    deviation = (reconstructed .- data) ./ data

    ### Prepare plots

    if fancy_plot == false # Original view (includes dates & times of day)

        # Display settings
        plot_size = (1920, 1920)
        dpi = 300

        # Set the layout for the plots
        l = @layout [ a b ; c ; d ; e ] # See https://docs.juliaplots.org/latest/layouts/

        # Plot the principal component scores for each dataset (3x3)
        p_scores = plot(transformed', title = "Scores")
        # Plot the principal components for each dataset (3x3)
        p_bases = plot(projection(model), title = "Bases")

        # Plot a heatmap of the reconstructed data, next to original data, using the same color scale
        p_orig = plot_heatmap(data_relative(data, log_first = log_first, log_after = log_after), symmetric = true, title = "Original")
        p_reconstructed = plot_heatmap(data_relative(reconstructed, log_first = log_first, log_after = log_after), symmetric = true, scale = p_orig[2], title = "Reconstructed") # Plot on the same scale, for easier comparison

        # Plot a heatmap of the deviation
        p_dev = plot_heatmap(data_relative(deviation, log_first = false), symmetric = true, title = "Devation Fraction")

        # Display the plots
        display(plot(p_scores, p_bases, p_dev[1], p_orig[1], p_reconstructed[1], layout = l, size = plot_size, dpi = dpi))

    else # Publication-quality

        # Display settings
        plot_size_1 = (750, 750)
        dpi_1 = 800
        plot_size_2 = (1000, 1200)
        dpi_2 = 600

        # Set the layout for the plots
        #l = @layout [ a b c ; d ; e ; f ] # See https://docs.juliaplots.org/latest/layouts/
        #l = @layout [ a b ; c ; d ; e ]
        l1 = @layout [ a b ; c d ]
        #l2 = @layout [ a ; b ; c; d e ]
        #l2 = @layout [ a ; b d ; c e ]
        l2 = @layout [ b ; c; d e ; a ]
        

        # Common-use plot information
        rel_times = round.(convert(Array{Float64}, Dates.value.(times - times[1]) ) / (1000.0 * 60 * 60); digits = 2) # Recall Julia is 1-based indexing
        
        ### Line plots

        # Plot the principal component scores for each dataset (3x3)
        #labels = reshape((i->"t"*string(i)).(1:num_princ_components), (1, num_princ_components))
        all_labels = ["First Principal Component", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth"]
        labels = reshape(all_labels[1:num_princ_components], (1, num_princ_components))
        legend_pos = :topleft
        if true #is_gain == false
            legend_pos = :bottomleft
        end
        p_scores = plot(rel_times, transformed', title = "Transformed Spectra", xlabel = "Time (hours)", ylabel = data_ylabel, labels = labels, legend = legend_pos )
        # Plot the principal components for each dataset (3x3)
        #labels = reshape((i->"w"*string(i)).(1:num_princ_components), (1, num_princ_components))
        all_labels = ["First Basis Vector", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth"]
        labels = reshape(all_labels[1:num_princ_components], (1, num_princ_components))
        p_bases = plot(f, projection(model), xaxis = :log, title = "Basis Spectra", xlabel = "Frequency (Hz)", ylabel = "Basis Vector (-)", labels = labels)
        # Show the reconstruction, with each additional principal component
        p_demo = plot(f, reconstruction_slices[1], title = "Reconstruction (xᵣ)", label = reconstruction_labels[1], xaxis = :log, xlabel = "Frequency (Hz)", ylabel = data_ylabel, linewidth = 2, linestyle = :dot, linecolor = :black, legend = false)
        p_residuals = plot(f, reconstruction_slices[1] - reconstruction_slices[1], title = "Residuals (x - xᵣ)", label = reconstruction_labels[1], xaxis = :log, xlabel = "Frequency (Hz)", ylabel = data_ylabel, legend = :bottomright, linewidth = 2, linestyle = :dot, linecolor = :black)
        for i in 2:(num_princ_components + 2)
            residuals = reconstruction_slices[1] - reconstruction_slices[i]
            #VAF = 1 - Statistics.var(residuals) / Statistics.var(reconstruction_slices[1] - reconstruction_slices[2])
                # Numerator: residuals (straightforward)
                # Denominator: Original data, MEAN SUBTRACTED
            #reconstruction_labels[i] = reconstruction_labels[i]  * " (" * string(100 * round(VAF, digits = 4)) * "%)"
            plot!(p_demo, f, reconstruction_slices[i], label = reconstruction_labels[i])
            plot!(p_residuals, f, residuals, label = reconstruction_labels[i])

            # This may be another method:
                # https://en.wikipedia.org/wiki/Exploratory_factor_analysis#Parallel_analysis
                # Or a Scree plot
                # Or the value of the eigenvalues for each eigenvector (property of the transformation itself)
                    # https://www.seas.upenn.edu/~ese224/slides/800_pca.pdf
                # Another good resource: http://alexhwilliams.info/itsneuronalblog/2016/03/27/pca/ 
        end

        # Display the plots
        main_plot = plot(p_scores, p_bases, p_demo, p_residuals, layout = l1, size = plot_size_1, dpi = dpi_1)
        display(main_plot)

        # Save vector graphic
        if save_fancy_figures
            if is_gain
                typename = "gain"
            else
                typename = "phase"
            end
            savefig(main_plot, "impedance_PCA_"  * typename * "_lines_" * name * ".pdf")
            #savefig(main_plot, "impedance_PCA_"  * typename * "_lines_" * name * ".png")
        end


        ### Spectrograms

        # Prepare the zoomed-in regions in advance
        zoom_lims = [[50, 70], [450, 500]]
        rectangle(x1, x2, y1, y2) = Plots.Shape([x1, x2, x2, x1], [y1, y1, y2, y2])
        f_zoom = [i for i in zoom_lims[1][1]:zoom_lims[1][2]] # Not actually frequencies, just indices
        rel_times_zoom = rel_times[zoom_lims[2][1]:zoom_lims[2][2]]
        rect = rectangle(rel_times_zoom[1], rel_times_zoom[end], f_zoom[1], f_zoom[end])

        # Set up labels correctly
        if is_gain
            if log_first
                addendum = "log₁₀(|Z| / |Z₀|)"
            else
                addendum = "Z - Z₀"
            end
        else
            addendum = "ϕ - ϕ₀ (°)"
        end

        # Plot a heatmap of the reconstructed data, next to original data, using the same color scale
        p_orig = plot_heatmap(data_relative(data, log_first = log_first, log_after = log_after), xlabels = rel_times, ylabels = f, symmetric = true, title = addendum)
        plot!(rect, linewidth = 2, fillalpha = 0, legend = false, linecolor = :red, widen = false)
        title!("Original (x)")
        p_reconstructed = plot_heatmap(data_relative(reconstructed, log_first = log_first, log_after = log_after), xlabels = rel_times, ylabels = f, symmetric = true, scale = p_orig[2], title = addendum) # Plot on the same scale, for easier comparison
        plot!(rect, linewidth = 2, fillalpha = 0, legend = false, linecolor = :blue, widen = false)
        title!("Reconstructed (xᵣ), n = " * string(num_princ_components))

        # Plot a heatmap of the deviation
        p_dev = plot_heatmap(data_relative(deviation, log_first = false), xlabels = rel_times, ylabels = f, symmetric = true, title = "(xᵣ - x) / x")
        title!("Residuals (Normalized)")

        # Plot the zoomed regions
        rect2 = rectangle(rel_times_zoom[1], rel_times_zoom[end], 1, length(f_zoom))
        data_zoom = data_relative(data, log_first = log_first, log_after = log_after)[zoom_lims[1][1]:zoom_lims[1][2], zoom_lims[2][1]:zoom_lims[2][2]]
        reconstructed_zoom = data_relative(reconstructed, log_first = log_first, log_after = log_after)[zoom_lims[1][1]:zoom_lims[1][2], zoom_lims[2][1]:zoom_lims[2][2]]
        p_orig_zoom = plot_heatmap(data_zoom, xlabels = rel_times_zoom, ylabels = f_zoom, title = addendum)#, scale = p_orig[2], symmetric = true)
        plot!(rect2, linewidth = 4, fillalpha = 0, legend = false, linecolor = :red, widen = false)
        ylabel!("")
        p_reconstructed_zoom = plot_heatmap(reconstructed_zoom, xlabels = rel_times_zoom, ylabels = f_zoom, title = addendum)#, scale = p_orig[2], symmetric = true)
        plot!(rect2, linewidth = 4, fillalpha = 0, legend = false, linecolor = :blue, widen = false)
        ylabel!("")

        # Display the plots
        #main_plot = plot(p_scores, p_bases, p_demo, p_dev[1], p_orig[1], p_reconstructed[1], layout = l, size = plot_size, dpi = dpi)
        #main_plot = plot(p_dev[1], p_orig[1], p_reconstructed[1], p_orig_zoom[1], p_reconstructed_zoom[1], layout = l2, size = plot_size_2, dpi = dpi_2)
        main_plot = plot(p_orig[1], p_reconstructed[1], p_orig_zoom[1], p_reconstructed_zoom[1], p_dev[1], layout = l2, size = plot_size_2, dpi = dpi_2)
        display(main_plot)

        # Save vector graphic
        if save_fancy_figures
            #savefig(main_plot, "impedance_PCA_"  * typename * "_spectrograms_" * name * ".pdf")
            savefig(main_plot, "impedance_PCA_"  * typename * "_spectrograms_" * name * ".png")
        end

    end



    return nothing

end


# Perform PCA
function PCA_analysis(data :: Array{T, 2} where T; visuals :: Bool = true, f :: Array{T, 1} where T = [], times = [], name :: String = "", log_first :: Bool = true, log_after :: Bool = false, num_princ_components :: Int = 3, fancy_plot = true, is_gain = true, subtract_mean = false, save_fancy_figures = false)#, visuals :: Bool, log_first :: Bool, log_after :: Bool, num_princ_components :: Int)
    # Data : 2D array of numeric data
    # visuals : whether or not to create plots
    # log_first, log_after : see data_relative()

    if subtract_mean
        data = deepcopy(data)
        dataset_mean = sum(data, dims = 2) / size(data, 2)
        data = data .- dataset_mean
    end

    # Perform PCA
    model = fit(PCA, data, pratio = 1, maxoutdim = num_princ_components)
        # NOTE: Because mean = :nothing (default), MultivariateStats.jl will compute and subtract the mean for you

    # Reconstruct data to verify
    transformed = transform(model, data)

    # Diagnose and plot (if desired)
    if visuals
        if is_gain
            data_ylabel = "Gain (Ω)"
        else
            data_ylabel = "Phase (°)"
        end
        PCA_plot(f, times, data .+ dataset_mean, model, transformed, log_first, log_after, name = name, fancy_plot = fancy_plot, data_ylabel = data_ylabel, is_gain = is_gain, dataset_mean_vector = dataset_mean, restore_mean = true, save_fancy_figures = save_fancy_figures)
    end

    # Store and return the salient results
    return PCA_results(model, transformed)

end


# Treat several datasets with the same transformation (as determined by their combined dataset)
function PCA_analysis(data :: Vector{Array{T, 2}} where T, names :: Array{String, 1}; visuals :: Bool = true, f :: Array{T, 1} where T = [], times = [], log_first :: Bool = true, log_after :: Bool = false, num_princ_components :: Int = 3, fancy_plots = true, is_gain = true, save_fancy_figures = false)

    # Calculation
    data_nomean = (d-> d .- sum(d, dims = 2) / size(d, 2) ).(data) # Subtract the average from each dataset - notice this allocates
        # Despite appearances, sum(d, dims = 2) means sum in the second dimension only, not in the first two dimensions
        # Consequently, the mean for each data set is (as it should be) a vector, not a scalar
        # The mean of each data set is subtracted separately
    PCA_comb = PCA_analysis(hcat(data_nomean...), visuals = false, num_princ_components = num_princ_components) # Calculate the principal components on the combined guy
        # This gets the transformation for the collective set of zero-mean data

    # Separate the combined transformed data back into individual datasets
    PCA_sep = similar(data_nomean) # You could reuse data_nomean, but just as the vector...its contents are differently sized than the output we're about to accommodate
    ind = 0
    for i in 1:length(data)

        # Populate
        s = size(data[i], 2)
        PCA_sep[i] = PCA_comb.transformed[1:end, ind+1:ind+s] # TODO maybe you could use @view to avoid allocating here? But that might not play well with my custom type. If you could use @view, it would be pretty magical.
        ind += s

        # Plot
        if visuals
            if is_gain
                data_ylabel = "Gain (Ω)"
            else
                data_ylabel = "Phase (°)"
            end
            #@show size(PCA_sep[i])
            #@show size(data[i])
            PCA_plot(f, times[i], data[i], PCA_comb.model, PCA_sep[i], log_first, log_after, restore_mean = true, name = names[i], fancy_plot = fancy_plots, data_ylabel = data_ylabel, is_gain = is_gain, save_fancy_figures = save_fancy_figures)
        end

    end

    # Return that
    return PCA_results_multi(PCA_comb.model, PCA_sep) # Use the vector-type PCA_results to keep track of the individual datasets

end

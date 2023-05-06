
## A couple of functions to view the results of collocation and shooting *after* the training has taken place

# NOTE: uses subscriptnumber from heatmaps.jl

function plot_training_progress(loss_history, training_times, data, training_data, best_parameters, model, internal_scale_t, internal_scale_u; num_gain = 2, num_phase = 2, time_unit = "hour", trim_at_end = 10, collocation = true, original_data = [], name = "unnamed")
    # Model should be NN_1() [passed without calling] from main.jl

    # Plot settings
    l = @layout [a; b c; d e] # ASSUMES four variables in u(t)
    plot_size = (1000, 1000)
    dpi = 600

    # Plot the loss
    p_loss = plot(loss_history[1:(length(loss_history)-trim_at_end)], yaxis = :log, legend = false, color = 2, framestyle = :box)#, linecolor = :firebrick2)#:darkorange)
        # The way I had it before, the callback would append the best parameter set every time you plotted it
        # Those should be trimmed off
    
    plot!(p_loss, [argmin(loss_history)], [minimum(loss_history)], markershape = :circle, markercolor = :black, markersize = 3)
    ylabel!(p_loss, "Objective")
    xlabel!(p_loss, "Training Iteration")

    # Get the predicted trajectory for the best parameters
        # This should include u0 as well, since it's not exactly u[0]; it's something that I was training
    
    # Re-scale the data for sake of prediction, to match what the neural network expects to receive
    t_d = training_times * 60 / internal_scale_t # Re-scaled training time
    u = data ./ internal_scale_u # Re-scaled training data u(t) - the input to the neural network

    # Get the neural network, populated with the best set of parameters
    NN = model() # Generate the neural network object that we will use
    _, re = Flux.destructure(NN)
    if collocation
        NN = re(best_parameters) # Restructures the neural network (assigns the best parameters to it)
    else
        u0 = best_parameters[1:size(data, 1)] # Co-optimized with the neural network parameters
        NN = re(best_parameters[(size(data, 1)+1):end]) # The first four parameters are u0
    end

    s = size(training_data, 2)
    if collocation
        
        # Just evaluate the neural network at each value of "data" (u(t))
        prediction = hcat((i->NN( @view u[:, i] )).(1:s)...)
            # NOTE: This "u" should be d_smoothed, not the raw (unsmoothed) PCA

        prediction = prediction .* internal_scale_u ./ internal_scale_t * 60
        #println(internal_scale_u ./ internal_scale_t * 60)
            # Return it to real-world (not normalized) units
            # The NN outputs an estimate for du/dt. Both u and t need to be re-scaled.

        # Do the same on the unsmoothed data
        u_original = original_data ./ internal_scale_u
        prediction_original = hcat((i->NN( @view u_original[:, i] )).(1:s)...)
        prediction_original = prediction_original .* internal_scale_u ./ internal_scale_t * 60

    else

        # There are convenient functions in learning.jl for this
        predictor = predict_generator_simple_neural_ODE(u, re, f_generator_simple_neural, u0, [t_d[1], t_d[end]], t_d, BacksolveAdjoint, ZygoteVJP())
        prediction = predictor(best_parameters)
        prediction = prediction .* internal_scale_u

    end

    # Plot the best result for each of the principal components
    p_individual = []
    for i in 1:(num_gain + num_phase)

        if collocation
            label = "Smoothed Derivative"
        else
            label = "Data"
        end

        if i == 1
            legend = :topleft
        else
            legend = false
        end

        # Training data
        if original_data == []
            p = plot(training_times, training_data[i, :], legend = legend, label = label, framestyle = :box)
                # Only show the legend for the first value, u1(t)
        else
            # Predicted trajectory, on original (unsmoothed) data - optional, and for collocation only
            p = plot(training_times, prediction_original[i, :], linecolor = :lightgray, label = "", framestyle = :box)# label = "Prediction (unsmoothed input)")#, order = :back)
                # It's here so that it shows in the back (z_order is broken)
            # ...then the training data.
            plot!(p, training_times, training_data[i, :], legend = legend, label = label, color = 1)
        end
        
        xlabel!("Time (" * time_unit * "s)")
        
        if i > num_gain
            if collocation
                ylabel!("du" * subscriptnumber(i) * "/dt (° / " * time_unit * ")")
            else
                ylabel!("u" * subscriptnumber(i) * " (°)") # Shooting
            end
            title!("Phase ϕ, Component " * string(i - num_gain))
        else
            if collocation
                ylabel!("du" * subscriptnumber(i) * "/dt (Ω / " * time_unit * ")")
            else
                ylabel!("u" * subscriptnumber(i) * " (Ω)") # Shooting
            end
            title!("Gain |Z|, Component " * string(i))
        end
        
        # Predicted trajectory
        plot!(training_times, prediction[i, :], label = "Prediction", linecolor = :black)



        push!(p_individual, p)
        


    end


    # Display the plot
    main_plot = plot(p_loss, p_individual[1], p_individual[2], p_individual[3], p_individual[4], layout = l, size = plot_size, dpi = dpi)
    display(main_plot)

    # Save vector graphic
    if collocation
        typename = "collocation"
    else
        typename = "shooting"
    end
    savefig(main_plot, "training_"  * typename * "_" * name * ".pdf")



end


############## NOTE: The functions below ASSUME the time_scale and scale (u_scale) that was used in Fall 2020


## A couple of functions to check how good your trained models *really* are

# Choose random data points from the real data and see what your model thinks they'll do
function check_trajectories_on_data(master_results, PCA_all, timings, names)

    for i in 1:length(master_results)

        res = master_results[i]

        # Get the data
        d = deepcopy(PCA_all[i])

        # Rescale the data to match what was trained on (TODO this needs to be ironed out later)
        scales = ( i->maximum(abs.(d[i, :])) ).(1:size(d, 1)) .* 1.25f0
        d = d ./ scales
        time_scale = 10f0
        t_d = deepcopy(timings[i]) ./ time_scale

        # The plotting code here is adapted from the learning callback

        # Plot the actual data
        s = 4
        plots = Vector{Plots.Plot}(undef, s)
        for k in 1:s
            plots[k] = plot(timings[i] ./ 60, d[k, :] .* scales[k], linewidth = 3)
        end

        # This is not strictly necessary, just an artifact of how stuff is called (TODO could be cleaned up)
        sensealg = BacksolveAdjoint
        autojacvec = ZygoteVJP()

        # Plot several sample trajectories extending from the dataset
        num_trajectories = 5
        num_gain = 2
        start_ind = Int64.(round.(range(1, stop = size(d, 2), length = num_trajectories+1)))
        start_ind = start_ind[1:end-1]
        for j in 1:num_trajectories

            # Choose a starting point and run the trajectory
            @show u0 = d[:, start_ind[j]] # Pluck a point directly from the data
            predict = predict_generator_simple_neural_ODE(d, res.re, f_generator_simple_neural, u0, [t_d[start_ind[j]], t_d[end]], t_d[start_ind[j]:end], sensealg, autojacvec)
            θ = deepcopy(res.θ_shooting)
            θ[1:4] .= u0
            prediction = predict(θ)

            # Contribute this trajectory to the plots
            for k in 1:4
                plot!(plots[k], timings[i][start_ind[j]:end] ./ 60, prediction[k, :] .* scales[k])#, linestyle = :dash)#, linewidth = 3)
                scatter!(plots[k], [timings[i][start_ind[j]]] ./ 60, [u0[k]] .* scales[k], ylims = (1.25 * minimum(d[k, :]) .* scales[k], 1.25 * maximum(d[k, :]) .* scales[k]), legend = false, markersize = 4, markercolor = :white, xlabel = "Time (hours)", framestyle = :box)#, ylabel = "Principal Component")
                if k > num_gain
                    ylabel!(plots[k], "u" * subscriptnumber(k) * " (°)")
                    title!(plots[k], "Phase ϕ, Component " * string(k - num_gain))
                else
                    ylabel!(plots[k], "u" * subscriptnumber(k) * " (Ω)")
                    title!(plots[k], "Gain |Z|, Component " * string(k))
                end
            end

        end

        # Create a plot layout
        layout = (@layout [ a b ; c d ])

        # Display all
        plot_size = (800, 800)
        dpi = 600
        main_plot = plot(plots..., layout = layout, size = plot_size, dpi = dpi)
        display(main_plot)

        # Save vector graphic
        savefig(main_plot, "check_on_trajectory_" * names[i] * ".pdf")

    end

end


# Perturb the initial conditions from those the NN found and see whether nearby solutions are similar
function check_trajectories_nearby(master_results, PCA_all, timings, names)

    for i in 1:length(master_results)

        res = master_results[i]

        # Get the data
        d = deepcopy(PCA_all[i])

        # Rescale the data to match what was trained on (TODO this needs to be ironed out later)
        scales = ( i->maximum(abs.(d[i, :])) ).(1:size(d, 1)) .* 1.25f0
        d = d ./ scales
        time_scale = 10f0
        t_d = deepcopy(timings[i]) ./ time_scale

        # The plotting code here is adapted from the learning callback

        # This is not strictly necessary, just an artifact of how stuff is called (TODO could be cleaned up)
        sensealg = BacksolveAdjoint
        autojacvec = ZygoteVJP()

        # Plot several sample trajectories extending from points originating near the trained u0
        plots = Vector{Plots.Plot}(undef, 4)
        num_trajectories = 8
        colors = range(colorant"gray", stop = colorant"firebrick1", length = num_trajectories - 1)
        num_gain = 2
        rand_mag = ones(num_trajectories)
        rand_mag[1] = 0.0
        for j in 1:num_trajectories

            # Generate a "nearby trajectory"
            θ = deepcopy(res.θ_shooting)
            θ[1:4] .= θ[1:4] .+ rand_mag[j] * 0.05 * (1.0 .- 2.0 * rand(4))
            predict = predict_generator_simple_neural_ODE(d, res.re, f_generator_simple_neural, θ[1:4], [t_d[1], t_d[end]], t_d, sensealg, autojacvec)
            prediction = predict(θ)

            # Contribute this trajectory to the plots
            for k in 1:4
                if j == 1

                    # Data
                    plots[k] = plot(timings[i] ./ 60, d[k, :] .* scales[k], linewidth = 3, color = 1)#, linecolor = :lightgray)
                    
                    # Predicted trajectory using the trained u0 value
                    plot!(plots[k], timings[i] ./ 60, prediction[k, :] .* scales[k], linewidth = 3, xlabel = "Time (hours)", linecolor = :black)#, color = 1)#, ylabel = "Principal Component")
                    
                    if k > num_gain
                        ylabel!(plots[k], "u" * subscriptnumber(k) * " (°)")
                        title!(plots[k], "Phase ϕ, Component " * string(k - num_gain))
                    else
                        ylabel!(plots[k], "u" * subscriptnumber(k) * " (Ω)")
                        title!(plots[k], "Gain |Z|, Component " * string(k))
                    end

                else
                    plot!(plots[k], timings[i] ./ 60, prediction[k, :] .* scales[k], ylims = (1.25 * minimum(d[k, :] .* scales[k]), 1.25 * maximum(d[k, :]) .* scales[k]), legend = false, framestyle = :box, linealpha = 0.5)#, linecolor = colors[j-1])#, linewidth = 0.5)#:firebrick2)#, linestyle = :dot) # , linestyle = :dot, linewidth = 3
                end

            end

        end

        # Create a plot layout
        layout = (@layout [ a b ; c d ])

        # Display all
        plot_size = (800, 800)
        dpi = 600
        main_plot = plot(plots..., layout = layout, size = plot_size, dpi = dpi)
        display(main_plot)

        # Save vector graphic
        savefig(main_plot, "check_nearby_" * names[i] * ".pdf")

    end

end

# Some extra utilities that specialize DiffEqFlux for our application, including necessary visualizations
# Mostly: loss (objective) functions and function generators; plotting callbacks
# NOTE: This file only contains the final versions. For naive implementations, see the benchmarking file(s).

using DifferentialEquations
using DiffEqSensitivity
using Flux
using DiffEqFlux
using Optim
using Zygote
using PlotThemes
using LinearAlgebra
BLAS.set_num_threads(1) # Disable multithreaded BLAS to allow for fully parallel analysis on separate datasets
using BenchmarkTools


## Objective function combiner

# TODO This is kinda not used right now but I could see it being used at some point
function Combine_Objectives(objectives)
    # Really simple utility: Say you need to combine several already-defined objective functions (e.g. to combine the loss from multiple datasets). This outputs that as a function of p only.
    return function objective(p) # Notice they all use the same parameter p
        obj = 0
        for i in 1:length(datasets)
            obj += objectives[i](p)
        end
        return obj
        #return sum((obj->obj(p)).(objectives)) TODO if I want a one-liner, then mapreduce would be the trick
    end
end


## Callbacks

function dont_forget_objective()
    println("You need to supply a loss function if you're using plot_best_of_batch = true in learning_callback_generator()!")
end

function learning_callback_generator(reference_data, t_d; verbose = false, visuals = true, plot_every = 300, plot_best_of_batch = true, loss_function = dont_forget_objective, stage_label = "", reference_label = "Reference Data", prediction_label = "Prediction", plot_size = (1000, 1000), create_layout = true, layout = nothing)
    # reference_data : the data we're trying to match during the learning process
    # t_d : the timestamps for that data
    # verbose : Do you wanna hear about every single loss calculation we perform? :)
    # visuals : plot?
    # plot_every : plot once every ____ iterations (helps for performance! Plotting is expensive.)
    # plot_best_of_batch : if plot_every > 1, then should we display the best (true) or most recent (false) of the iterations since the last plot?
    # loss_function : if you have plot_best_of_batch, this is required to reconstruct the prediction from a past parameter value. Set it to your loss function.
    # force_plot : ignore both visuals and plot_every to, guaranteed, produce a plot
    # Some text labels for the plots:
        # stage_label : e.g. "Collocation" or "Shooting", for the loss (objective) graph
        # reference_label : what should we call the reference data?
        # prediction_label : what should we call the prediction?
    # plot_size : the size parameter for the plotting call (canvas size for the plot)
    # create_layout : true if you don't supply a plot layout. If you want to supply a layout, set it to false, or else your layout will be overwritten.
    # layout : a plotting layout appropriate for your data (should have one entry for the objective plot, and N entries for the N variables of your system). The default is for a four-variable system. See https://docs.juliaplots.org/latest/layouts/

    loss_trend = Vector{Float64}()
    p_history = Vector{Vector{Float32}}()
    plot_ind = [1] # Keep track of how many iterations it's been since we've plotted. Vector so that it is stored on the heap.

    callback = function (p, loss_value, prediction; force_plot = false)

        # Display text in REPL
        if verbose
            display(loss_value)
        end
        if plot_ind[1]%plot_every == 0
            println("Round " * string(plot_ind[1]))
        end

        # Record values
        append!(loss_trend, loss_value)
        append!(p_history, [deepcopy(p)])

        # Do some plotting
        if force_plot || (visuals && plot_ind[1]%plot_every == 0)
            #display("I've been asked to plot")

            # Change behavior depending on preferences
            if plot_best_of_batch && (force_plot == false)
                # Go back and find the best fit in the last plot_every iterations
                l = length(loss_trend)
                best_index = argmin(@view(loss_trend[ l-plot_every+1 : l ])) + l - plot_every
                best_loss, plot_prediction = loss_function(p_history[best_index])
                #@show best_loss
                #@show size(plot_prediction)
                #@show plot_prediction[1, 1]
            else
                plot_prediction = prediction
            end

            # Generate the loss plot
            loss_plot = plot(loss_trend, yaxis = :log, label = stage_label * " Objective Value")
            if plot_best_of_batch && (force_plot == false)
                scatter!([best_index], [best_loss], label = "Plotted Below") # Label the point that we're elaborating on
            end

            # Generate the estimators
            s = size(plot_prediction, 1)
            plots = Vector{Plots.Plot}(undef, s)
            for i in 1:s
                plots[i] = plot(t_d, reference_data[i, :], label = reference_label)
                plot!(plots[i], t_d, prediction[i, :], label = prediction_label, legend = :bottomright)
            end

            # Create a plot layout if one hasn't been provided
            if create_layout
                layout = (@layout [ a ; b c ; d e ])
            end

            # Display all
            display(plot(loss_plot, plots..., layout = layout, size = plot_size))
        end

        # Keep an external iterator going, so we can plot every few steps instead of every step; speeds things up a lot
        plot_ind[1] += 1

        return false # If it returns true, optimization stops; if you don't have a "return" statement, it just returns some random result from the last line
    end

    # Return the psuedo-global variables as well as the actual function
    # The variables will be mutated and contain results. Similarly, you can manipulate them.
    return callback, loss_trend, p_history, plot_ind

end



## Systems

# Try regularizations etc.

# Define the different types of ODE systems that might be useful to approximate the evolution of principal components
# Apart from the utilities, each of these defines f(u, p, t) = du/dt for the principal component scores u (or defines utilities for generating such functions)


## Objective functions
# Should take in a function which is the ODE, its parameters, maybe some solution options (or those could be handled by an ODEProblem that gets passed in), and the data, and output a nicely-wrapped objective function for Flux or DiffEqFlux

# Function generator for the loss in collocation
function loss_generator_collocation(d_smoothed, dd_smoothed, f)
    # d_smoothed : smoothed data for collocation
    # dd_smoothed : derivative values of d_smoothed, of same size as d_smoothed
    # f : f(u, p, t) for your ODE

    return function (p)
        s = size(dd_smoothed, 2)
        predicted_du = hcat((i->f( (@view d_smoothed[:, i]), p, 0)).(1:s)...) # Hopefully it's type stable now, so fewer allocations?
        #loss_weight = 1 .+ 20exp.(-(1:s)/80) # Loss weight 21 at start, decaying over a time scale of 80 samples down to 1 eventually
        loss_weight = 1f0 .+ 20f0 * exp.(-(1:s)./80f0) # Loss weight 21 at start, decaying over a time scale of 80 samples down to 1 eventually
        loss = sum( (dd_smoothed .- predicted_du).^2 * loss_weight ) # There's a matrix multiplication in there - woah, why is the Zygote gradient faster now??
        return loss, predicted_du
    end

end

# Objective function for collocation
function loss_collocation(p, f_generator, d, dd, re)

        # Generate type-stable functions for evaluation
        f = f_generator(re, p) # 5.7 microseconds, 75 allocations (presumably all from reconstruct())
        loss = loss_generator_collocation(d, dd, f) # 88 nanoseconds, 1 allocation

        # Calculate the actual loss
        return loss(p)
end

# Function generator for the predictions used in the shooting method
#function predict_generator_simple_neural_ODE(d, re, f_generator, u0, tspan, t_d)

    # Set up the ODE solver
#    return function (θ)
#        s = size(d, 1)
#        u0 = @inbounds @view(θ[1:s])
#        p = @inbounds @view(θ[s+1:end])
#        f = f_generator(re, p)
#        problem = ODEProblem(f, u0, tspan)
#        return Array(solve(problem, Tsit5(), saveat = t_d)) # TODO saveat is probably slower than the interpolation provided by solve()!
#    end

#end

# Function generator for the predictions used in the shooting method
    # NOTE This version is specifically for benchmarking; if any of these are good, should delete and make them optional arguments with sensible defaults
    # sensealg : BacksolveAdjoint(), InterpolatingAdjoint(), QuadratureAdjoint()
    # autojacvec : ReverseDiffVJP(compile = true), ZygoteVJP()
function predict_generator_simple_neural_ODE(d, re, f_generator, u0, tspan, t_d, sensealg, autojacvec)

    # Set up the ODE solver
    return function (θ)
        s = size(d, 1)
        u0 = @inbounds @view(θ[1:s])
        p = @inbounds @view(θ[s+1:end])
        f = f_generator(re, p)
        problem = ODEProblem(f, u0, tspan, sensealg = sensealg(autojacvec = autojacvec))
        return Array(solve(problem, Tsit5(), saveat = t_d)) # TODO saveat is probably slower than the interpolation provided by solve()!
    end

end

# Objective function for shooting; note this does NOT internally generate the prediction function
function loss_shooting(θ, predict, d)
    prediction = predict(θ)
    loss = sum(abs2, d .- prediction) # This subtracts the actual from predicted trajectory, then squares all the resulting elements and adds them together (basically L2 loss)
    return loss, prediction # Could omit the "return" but that's too much of a Julia-ism for me
end



## Neural (of some description)
# All of the f_generator's should take (re, p) to reconstruct the neural network (or whatever other structure they use) from p

# (ODE Nonlinear) Neural with no time history
    # (This is largely to test / understand the neural network framework, and to prove a point that you can't get very good fits without using time-history because it's not f(u, p) but f(u1, u2, ..., p) )

function f_generator_simple_neural(re, p)
    # Returns a simple f(u, p, t) = NN(u)
    # re should be the reconstructing function from Flux.destructure
    # p should be the parameters that re uses to reconstruct the neural network

    rec = re(p)
    return function (u, p, t)
        rec(u)
    end

end


# A compact way of storing fairly rich output during multithreaded training
struct TrainingResults{T}

    # Functions to return
    re :: Any
    callback_collocation :: Any
    callback_shooting :: Any

    # Numeric values
    p_collocation :: Vector{T}
    θ_shooting :: Vector{T}
    losses_collocation :: Vector{T}
    losses_shooting :: Vector{T}
    p_history_collocation :: Vector{Vector{T}}
    θ_history_shooting :: Vector{Vector{T}}

end


function automated_train_simple_neural!(NN_gen, d, t_d;
                                        time_scale = 10f0,
                                        average_until = 10,
                                        visuals = true,
                                        iters_collocation = 5000, iters_shooting = 2000, iters_shooting_2 = 3000,
                                        learnrate_collocation = 0.05f0, learnrate_shooting = 0.001f0, learnrate_shooting_2 = 0.00001f0,
                                        attempts_collocation = 3, attempts_shooting = 1,
                                        sensealg = BacksolveAdjoint, autojacvec = ZygoteVJP())#autojacvec = ReverseDiffVJP(true))

    # NN_gen : a function which takes no input arguments, and spits out an instantiated neural network
        # Note: This function enforces Float32 type conversion internally; make sure your neural network provides this (should automatically)
        # Ensure all things meant to be ints are ints
    # d : the data to fit it to (gets mutated)
    # t_d : the timestamps for that data (gets mutated)
    # time_scale : a constant to scale (divide) the time by before training
    # average_until : how many initial timestamps to take for your u0 guess
        # NOTE : This in particular should be an integer!!
    # visuals : whether to show plots automatically

    println("Entered training.")

    #### Sanitize the input data so everything is Float32 (to avoid type conversion internally)
        # NOTE : The supplied neural network should also be FLoat32!

    d = Float32.(d)
    t_d = Float32.(t_d)

    #### Set up and scale the data

    # Scale the data so it's easier to learn (everything of roughly same magnitude)
    scales = ( i->maximum(abs.(d[i, :])) ).(1:size(d, 1)) .* 1.25f0 # Important: make sure this is a vector (column vector)
    d = d ./ scales # This divides each row by the corresponding element in the scales vector; does not mutate PCA_all
        # Data will now be strictly bounded between -1 and 1, and will never hit either

    # Average the first several measurements to get a reasonably noise-free initial condition estimate
    #average_until = 10
    u0 = reshape(sum(d[:, 1:average_until], dims = 2) ./ average_until, size(d, 1)) # The reshape is to make sure this is a vector, not an Nx1 Array

    # Get timing parameters
    #time_scale = 10
    t_d .= t_d ./ time_scale # This helps the neural network decide how much to "wiggle"
    tspan = [t_d[1], t_d[end]]

    # Decide which f(u, p, t) generator you want to use
    f_generator = f_generator_simple_neural

    #### Smoothed data for collocation

    # Smooth out the data
    println("Type of d is now " * string(eltype(d)))
    d_smoothed = smooth(d, visuals = visuals)

    # Calculate the derivative
    # NOTE : Normally in collocation you would use a fancy interpolation, but that didn't work. So instead, we'll just use boring numerical analysis techniques.
    dd_smoothed = finite_difference(d_smoothed, t_d[2] - t_d[1])
    if visuals
        plot(dd_smoothed')
    end


    ############## BEGIN SUBFUNCTIONS
    # These inherit information from the above, so have to be positioned here


    # Collocation stage
    function train_collocation(NN)
        # This will be run multiple times, the only change being the NN; all other data is fine to inherit

        # Get instructions for how to reconstruct the neural network
        p, re = Flux.destructure(NN) # Gets p as a vector; re is the function to restructure the NN
            # Note that re(p) does not mutate NN; it returns a new NN

        # Choose the loss function for collocation
        collocation_loss_choice(p) = loss_collocation(p, f_generator, d_smoothed, dd_smoothed, re)

        # Define the callback, which will store information, print lines to the REPL, and plot your progress!
        callback_collocation, loss_history_collocation, p_history_collocation, _ = learning_callback_generator(dd_smoothed, t_d,
                                                                                                               loss_function = collocation_loss_choice,
                                                                                                               stage_label = "Collocation",
                                                                                                               reference_label = "Smoothed Derivative",
                                                                                                               prediction_label = "NN Output",
                                                                                                               visuals = visuals)
        # Plot before
        callback_collocation(p, collocation_loss_choice(p)..., force_plot = visuals);
            # Here's a fancy Juliaism: use the multiple return arguments from loss() as separate input arguments...

        # Perform the training
        result_collocation = DiffEqFlux.sciml_train(collocation_loss_choice, p,
                                                    ADAM(learnrate_collocation),#ADAM(0.05),
                                                    maxiters = iters_collocation,
                                                    cb = callback_collocation)
            # This is fast now! It's down to 3.2 seconds per 100 iterations, so under 3 minutes for 5000 iterations
            # Old message: This cranks along at 7.4 seconds per 100 iterations, so a little over 6 minutes for 5000 iterations.
            # Shoot, all that work and it's not really that much faster. I guess the key is to remember: the GRADIENT, not the function itself, is the expensive part.
            # It was *an* improvement, but not a big enough one to spend several hours on....

        # Plot afterwards
        p_opt = p_history_collocation[argmin(loss_history_collocation)]
            # Here, you can manually exclude bad regions just by taking a subset of the whole loss_trend, so you can find what YOU think was the best fit, not the program
        callback_collocation(p_opt, collocation_loss_choice(p_opt)..., force_plot = visuals)

        return p_opt, re, loss_history_collocation, p_history_collocation, p -> callback_collocation(p, collocation_loss_choice(p)..., force_plot = true)

    end


    # Shooting stage
    function train_shooting(re, θ)
        # This will be run multiple times, the only change being the NN reconstructor and the starting parameters θ; all other data is fine to inherit

        # Set up the prediction that we'll use to solve the ODE
        predict = predict_generator_simple_neural_ODE(d, re, f_generator, u0, tspan, t_d, sensealg, autojacvec)

        # Set up the loss that we'll use in training
        loss = θ -> loss_shooting(θ, predict, d)

        # Define the callback, which will store information, print lines to the REPL, and plot your progress!
        callback_shooting, loss_history_shooting, θ_history_shooting, _ = learning_callback_generator(d, t_d,
                                                                                                      loss_function = loss,
                                                                                                      stage_label = "Shooting",
                                                                                                      prediction_label = "ODE Prediction",
                                                                                                      visuals = visuals)

        # Display our starting point
        callback_shooting(θ, loss(θ)..., force_plot = visuals)

        # FIRST STEP training
        result_ode = DiffEqFlux.sciml_train(loss, θ,
                                            ADAM(learnrate_shooting),
                                            cb = callback_shooting,
                                            maxiters = iters_shooting)

        # Harvest and show the results of the first step
        θ_opt = θ_history_shooting[argmin(loss_history_shooting)] # The nice thing: here, you can manually exclude bad regions just by taking a subset of the whole loss_trend, so you can find what YOU think was the best fit, not the program
        callback_shooting(θ_opt, loss(θ_opt)..., force_plot = visuals)

        # SECOND STEP training
        # Note we use the same callback - so the results get appended to the results of the previous training
        result_ode_2 = DiffEqFlux.sciml_train(loss, θ_opt,
                                              ADAM(learnrate_shooting_2),#ADAM(0.0005),
                                              cb = callback_shooting,
                                              maxiters = iters_shooting_2)

        # Harvest and show the results of the second step
            # I'm overwriting those of the first step because I can' tbe bothered to do otherwise
        θ_opt = θ_history_shooting[argmin(loss_history_shooting)]
        callback_shooting(θ_opt, loss(θ_opt)..., force_plot = visuals)

        #return θ_opt_1, θ_opt_2, loss_history_shooting, θ_history_shooting
        return θ_opt, loss_history_shooting, θ_history_shooting, θ -> callback_shooting(θ, loss(θ)..., force_plot = true)

    end

    ##### END SUBFUNCTIONS

    # Begin actual training and logic

    # Train the collocation step
    p_collocated = []
    re_collocated = []
    losses_collocated = [] # Vector of scalars
    cb_collocated = []
    p_hist_coll = []
    for i in 1:attempts_collocation # Make several attempts
        println("Starting a collocation round")
        p_opt, re, loss_history_coll, p_history_coll, cb_coll = train_collocation(NN_gen()) # Tries it each time with a fresh neural network
        append!(p_collocated, [p_opt]) # Merges it in (like python "extend") if p_opt isn't in a vector
        append!(re_collocated, [re]) # Complains if re isn't in a vector
        append!(losses_collocated, loss_history_coll[end]) # The "end" call is okay because of the final callback call at the end of train_collocation
        append!(cb_collocated, [cb_coll])
        append!(p_hist_coll, [p_history_coll])
    end
    i_opt = argmin(losses_collocated)
    p_opt = p_collocated[i_opt]
    re_opt = re_collocated[i_opt]
    cb_coll_opt = cb_collocated[i_opt]
    p_hist_coll_opt = p_hist_coll[i_opt]

    # Protect the optimized result by making a deepcopy before playing further
    p = deepcopy(p_opt)
    θ = [u0; p] # Recreate the θ combined u0 / p parameter

    # Train the shooting step
    θ1 = [] # Best parameters after first stage
    #θ2 = [] # Best parameters after second stage (could be same as first)
    losses_shooting = [] # Vector of scalars
    cb_shooting = []
    θ_hist_shooting = []
    for i in 1:attempts_shooting # Make several attempts
        println("Starting a shooting round")
        #θ_opt_1, θ_opt_2, loss_history_shooting, _ = train_shooting(re_opt, θ)
        θ_opt_1, loss_history_shooting, θ_history_shooting, cb_shoot = train_shooting(re_opt, θ)
        append!(θ1, [θ_opt_1])
        #append!(θ2, [θ_opt_2])
        append!(losses_shooting, loss_history_shooting[end]) # The "end" call is okay because of the final callback call at the end of train_collocation
        append!(cb_shooting, [cb_shoot])
        append!(θ_hist_shooting, [θ_history_shooting])
    end
    i_opt = argmin(losses_shooting)
    θ_opt_1 = θ1[i_opt]
    cb_shoot_opt = cb_shooting[i_opt]
    θ_hist_shoot_opt = θ_hist_shooting[i_opt]
    #θ_opt_2 = θ2[argmin(losses_shooting)]

    println("Training complete.")

    # Return the most salient results
    #return re_opt, p_opt, θ_opt_1, losses_collocated, losses_shooting, cb_coll_opt, cb_shoot_opt
    return TrainingResults{Float32}(re_opt, cb_coll_opt, cb_shoot_opt, p_opt, θ_opt_1, losses_collocated, losses_shooting, p_hist_coll_opt, θ_hist_shoot_opt)

end




function benchmark_adjoint!(NN_gen, d, t_d, sensealg, autojacvec; time_scale = 10f0, average_until = 10)
    # Same input arguments as the above, except with the sensealg and autojacvec specified externally

    println("Entered benchmark: " * string(sensealg) * "; " * string(autojacvec))

    #### Sanitize the input data so everything is Float32 (to avoid type conversion internally)
        # NOTE : The supplied neural network should also be FLoat32!

    d = Float32.(d)
    t_d = Float32.(t_d)

    #### Set up and scale the data

    # Scale the data so it's easier to learn (everything of roughly same magnitude)
    scales = ( i->maximum(abs.(d[i, :])) ).(1:size(d, 1)) .* 1.25f0 # Important: make sure this is a vector (column vector)
    d = d ./ scales # This divides each row by the corresponding element in the scales vector; does not mutate PCA_all
        # Data will now be strictly bounded between -1 and 1, and will never hit either

    # Average the first several measurements to get a reasonably noise-free initial condition estimate
    u0 = reshape(sum(d[:, 1:average_until], dims = 2) ./ average_until, size(d, 1)) # The reshape is to make sure this is a vector, not an Nx1 Array

    # Get timing parameters
    t_d .= t_d ./ time_scale # This helps the neural network decide how much to "wiggle"
    tspan = [t_d[1], t_d[end]]

    # Decide which f(u, p, t) generator you want to use
    f_generator = f_generator_simple_neural

    ##### Prep the NN

    NN = NN_gen()
    # Get instructions for how to reconstruct the neural network
    p, re = Flux.destructure(NN) # Gets p as a vector; re is the function to restructure the NN
        # Note that re(p) does not mutate NN; it returns a new NN
    θ = [u0; p] # Recreate the θ combined u0 / p parameter

    ##### Do the benchmark

    # Set up the prediction that we'll use to solve the ODE
    predict = predict_generator_simple_neural_ODE(d, re, f_generator, u0, tspan, t_d, sensealg, autojacvec)

    # Set up the loss that we'll use in training
    function loss(θ)
        l, _ = loss_shooting(θ, predict, d)
        return l
    end

    # Benchmark
    #loss(θ)
    #@time Zygote.gradient(loss, θ)
    return loss, θ # Benchmark externally to support @btime

end



# NOTE (for sometime later) : Lecture 15 has other neural ODE formulations (physics-informed) that might make sense on further study



# (DDE Nonlinear) Neural with time history
    # How much can you squeeze it?
    # How many layers do you need?
# function DDE_neural()


# (Coarse DDE Nonlinear) Neural with time history with hierarchical first weight layer



## Volterra

# (DDE Linear) First-order


# (DDE Nonlinear) Second-order

    # Main function


    # Reshape parameters into Volterra kernel(s)...
        # If dense
        # If hierarchical



## Extrapolation (finite differencing -> predictions) ... ?

# Linear, based on local derivative
    # (This is mostly just to prove a point)

# Some sort of curvy extrapolation, like Taylor or something...



## SINDy ... ?






## Utilities
# Here should be conveniences, for stuff like comparing different objective functions, different adjoints (from blog post, sounds like ForwardDiff, ReverseDiff, Zygote, and the "sensitivity equations" are all options, along with reverse-mode variants maybe...), different optimization algorithms (ADAM seems common)
# NOTE: You don't have to try every adjoint with every ODE system. Try a bunch with one of the systems, and then assume it holds for the others, or check strategically which ones might improve.
    # NOTE: Would actually be good to contrast the small-parameter systems with large-parameter systems. < 100, ForwardDiff should win, > 100 the various reverse modes will compete (see blog post)

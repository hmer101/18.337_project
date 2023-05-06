
## TEMP - should go in module.jl when time comes, along with all those "export" commands

using Base.Threads
println("Worker threads available: " * string(nthreads()))

include("read_files.jl");
include("impedance.jl");
include("heatmaps.jl");
include("PCA.jl");
include("numerics.jl");
include("learning.jl");
include("post_training.jl")

# Set the plot theme
theme(:default) # To prevent seizures while watching the learning update plots flash in and out
    # Dark: use :juno or similar
    # Light: you've got some options...
        # :default is fine
        # I prefer the default line colors from wong2, vibrant or dao, but they have other issues.
    # https://docs.juliaplots.org/latest/generated/plotthemes/

#plotlyjs()
    # For interactive, vector plots
    # Breaks compatibility with packages in learning.jl
    # Also has issues with multiple legends in subplots - fundamental problem with plotly / plotlyjs

#using PyPlot
#pygui(true)
    # For interactive, vector plots via matplotlib (sends it straight to Python)

#Plots.pyplot()
    # For interactive, vector plots via matplotlib (sends it straight to Python)

##  TEMP - A little script - which should go in the main script file, outside of the package


## Load data

experiment = "Experiment_1"

# Load the data TODO - parallelize?
files = find_files(pwd() * "\\" * experiment * "\\Ordered_Data")
    # This pwd() expects your working directory to be the Project_I folder (or whichever Project_x folder you're on by now)

# Trim by variable amount depending which dataset we're looking at
if experiment == "Experiment_1"
    trim_by = [(30, 5), (1, 1)] # NOTE: Trims the first and last frequency
elseif experiment == "Experiment_2"
    trim_by = [(1000, 20000), (1, 1)] # NOTE: Trims the first and last frequency
else
    println("No trimming rule specified for this experiment. Automatically trimming first and last frequency bin.")
    trim_by = [(0, 0), (1, 1)] # Reasonable default
end

data, f = assemble_data(files, exclude = ["settings", "frequencies"], trim_data_by = trim_by)
    # TODO This is using a lot of CPU and some memory, but not very much disk at all!
    # Probably has something to do with how I'm handling the data...

# Exclude any broken datasets
if experiment == "Experiment_1"
    #deleteat!(data, 2:9) # TEMPORARY TEMPORARY JUST FOR TESTING
    deleteat!(data, 2:3)
    deleteat!(data, 5) # This one had the clips start shifting (I assume that's what it was) further down the line
elseif experiment == "Experiment_2"
    println("No exclusions for experiment 2.")
else
    println("No data specified to exclude from this experiment.")
    # Reasonable default is not to delete anything
end


## Calculate impedances and check timing

# Calculate impedance
impedances = calculate_impedance(f, data, visuals = false)
    # TODO The plotting is bottlenecking. Could maybe do multithread, but it's questionable...execution order, which has control over plot window, etc. Maybe return all the plots, then generate them in a final call? In any case, not worth it.
 
# Analyze measurement time spacing for each dataset (time between measurements)
timing_deviation = check_time_spacing.(impedances)
timings = get_relative_times.(impedances) / 60000 # In minutes, after conversion
println("Timing standard deviation as fraction of measurement time: " * string(timing_deviation))
    # "timings" IS REUSED later on

## Principal component analysis (PCA)
PCA_visuals = true
save_fancy_figures = true
log_first = false # for gain only

# Perform analysis on each sample individually (for a check to ensure similar behavior before proceeding)
analyze_individuals = false
print(typeof(f))
if analyze_individuals
    # Perform PCA on each dataset's |Z| gain (independently)
    Z_PCA = (i->PCA_analysis(i.Z, f = f, times = i.times, name = i.name, subtract_mean = true, save_fancy_figures = save_fancy_figures, visuals = PCA_visuals, log_first = log_first)).(impedances) # (This way avoids allocating two lists, as opposed to that at right) #PCA_analysis.((i->i.Z).(impedances))
    # Perform PCA on each dataset's ϕ phase (independently)
    ϕ_PCA = (i->PCA_analysis(i.ϕ .* 180 / (2π), f = f, times = i.times, name = i.name, log_first = false, num_princ_components = 2, is_gain = false, subtract_mean = true, save_fancy_figures = save_fancy_figures, visuals = PCA_visuals)).(impedances)
end

# Perform PCA on the whole set's Z-impedance (collectively)
Z_PCA_all = PCA_analysis((i->i.Z).(impedances), (i->i.name*"_all").(impedances), f = f, times = (i->i.times).(impedances), num_princ_components = 5, visuals = PCA_visuals, save_fancy_figures = save_fancy_figures, log_first = log_first)
    # Note: "names" is mandatory for the batch analysis

# Perform PCA on the whole set's ϕ phase (collectively)
ϕ_PCA_all = PCA_analysis((i->i.ϕ .* 180 / (2π)).(impedances), (i->i.name*"_all").(impedances), f = f, times = (i->i.times).(impedances), num_princ_components = 5, log_first = false, visuals = PCA_visuals, is_gain = false, save_fancy_figures = save_fancy_figures)

# Combine the PCA'd Z and ϕ data so that you have four parameters to machine learn from - gives you a fighting chance with the instantaneous (no time-history) approaches
PCA_all = (j->vcat(Z_PCA_all.transformed[j], ϕ_PCA_all.transformed[j])).(1:length(impedances))
    # This is just a vector of Arrays, since the PCA model isn't super relevant to the ODE


## Simple neural ODE: Define a neural network

# Make a simple three-layer NN for instantaneous ODE
# (This function spits out neural networks on demand; helps with trying multiple initial parameters)
function NN_1()
    return Chain( # TODO FastChain might be better for reverse mode? (Despite being slower for forward mode.)
        Dense(4, 16, tanh), # Take 4 input principal components, rescale and "rotate", turn into a hidden layer of 4 components
        Dense(16, 16, tanh), # Do your magic on the hidden layer to predict their evolution
        Dense(16, 4)     # Turn it back into the principal component space (with no activation function, because your principal components are not bound to be less than 1!)
    ) # More basics: https://fluxml.ai/Flux.jl/stable/models/basics/
end


## Train the simple neural ODE automatically several times

# Force precompile by running it once
i = 1

d = deepcopy(PCA_all[i][:, 1:end])#truncate_at] # Some single dataset, combined Z / phi as above
t_d = deepcopy(timings[i][1:end])#truncate_at] # The converted time stamps for that dataset
# NOTE: d, t_d get mutated, so just chuck the expression they're equal to into the function

println("Thread " * string(i) * " about to start training")
res = automated_train_simple_neural!(NN_1, d, t_d, visuals = true, iters_collocation = 2, iters_shooting = 2, attempts_collocation = 1, attempts_shooting = 1) # Can't plot when multithreaded
# TODO TODO TODO : Need to scale this back to normal before proceeding (multiply by d's element-wise scale, divide by t_scale)
println("Thread " * string(i) * " has finished training.")

# Save the results
using JLD2
JLD2.@save "training_" * string(Dates.format(now(), "yyyy-mm-dd_HH-MM")) * ".jld2" res

# Generate many multithreaded training results (NOTE : This takes ~ 6 hours)
for j in 1:10
    # Do multithreaded training
    # NOTE : It takes 24 minutes to run with 3 / 1 attempts and 5000 iters each
    results = Vector{TrainingResults{Float32}}(undef, length(PCA_all))
    @time @threads for i in 1:6 # Static schedule multithreading
        # Establish the data to be used
        #truncate_at = 694
        d = deepcopy(PCA_all[i][:, 1:end])#truncate_at] # Some single dataset, combined Z / phi as above
        t_d = deepcopy(timings[i][1:end])#truncate_at] # The converted time stamps for that dataset
        # NOTE: d, t_d get mutated, so just chuck the expression they're equal to into the function

        println("Thread " * string(i) * " about to start training")
        res = automated_train_simple_neural!(NN_1, d, t_d, visuals = false)#, iters_collocation = 5, iters_shooting = 5, iters_shooting_2 = 5, attempts_collocation = 3, attempts_shooting = 1) # Can't plot when multithreaded
        # TODO TODO TODO : Need to scale this back to normal before proceeding (multiply by d's element-wise scale, divide by t_scale)
        println("Thread " * string(i) * " has finished training.")

        # Save the results to a vector in a thread-safe way
        results[i] = res

    end

    # Plot results from training
    for i in 1:length(results)
        res = results[i]
        res.callback_collocation(res.p_collocation)
        res.callback_shooting(res.θ_shooting)
    end

    # Save the results to disk
    using JLD2
    JLD2.@save "training_" * string(Dates.format(now(), "yyyy-mm-dd_HH-MM")) * ".jld2" results

end


## Read back the results for visual inspection

fnames = [
    "training_2020-12-18_01-50.jld2",
    "training_2020-12-18_02-23.jld2",
    "training_2020-12-18_03-11.jld2",
    "training_2020-12-18_03-48.jld2",
    "training_2020-12-18_04-35.jld2",
    "training_2020-12-18_05-14.jld2",
    "training_2020-12-18_06-00.jld2",
    "training_2020-12-18_06-46.jld2",
    "training_2020-12-18_07-36.jld2",
    "training_2020-12-18_08-33.jld2"
]

# Loop manually through the training sets one by one, so it's easier to handle
i = 0
begin

    i += 1

    # Load the results of a multithreaded training session
    JLD2.@load fnames[i] results

    #println(typeof(results))
    #println(typeof(results[1]))
    #println(results[1].callback_collocation)
    #println(typeof(results[1].callback_collocation))

    # Plot results from training
    for j in 1:length(results)

        res = results[j]

        check_integrity = true
        if check_integrity && j == 1
            println("p_collocation: " * string(typeof(res.p_collocation)))
            println("θ_shooting: " * string(typeof(res.θ_shooting)))
            println("losses_collocation: " * string(typeof(res.losses_collocation)))
            println("losses_shooting: " * string(typeof(res.losses_shooting)))
            println("p_history_collocation: " * string(typeof(res.p_history_collocation)))
            println("θ_history_shooting: " * string(typeof(res.θ_history_shooting)))
            println("callback_collocation.collocation_loss_choice.d_smoothed: " * string(typeof(res.callback_collocation.collocation_loss_choice.d_smoothed)))
        end

        #res.callback_collocation(res.p_collocation)
        #res.callback_shooting(res.θ_shooting)

    end

end


## Manually select the best approximations for further analysis

# Retrieve the best fits from saved data
best_fits = [9, 10, 8, 2, 1, 10] # One entry per bucket, each number corresponding to an index in fnames
master_results = similar(results)
for i in 1:length(best_fits)

    # Load the results of a multithreaded training session
    JLD2.@load fnames[best_fits[i]] results
    master_results[i] = results[i]

end

# Plot the dream team for the love and adoration of the masses
for j in 1:length(master_results)
    res = master_results[j]
    res.callback_collocation(res.p_collocation)
    res.callback_shooting(res.θ_shooting)
end



## Let's see whether these approximated ODEs can continue to give good results at intermediate starting points in the data

check_trajectories_on_data(master_results, PCA_all, timings, (i->i.name).(impedances))
    # Conclusion: Some of them are reasonably accurate! Others just aren't.


## Let's see whether nearby trajectories converge (are stable) or diverge (are chaotic)

check_trajectories_nearby(master_results, PCA_all, timings, (i->i.name).(impedances))
    # Conclusion: These are *really* sensitive to noise, and plainly could not be applied to samples other than those they were trained on



####################################


# Recovering old data

for i in 1:length(impedances)

    # ASSUME that you have loaded the dataset using the above script that matches the data you just loaded in
    res = master_results[i]
    imp = impedances[i]
    pca = PCA_all[i]


    #### COLLOCATION

    # Some of the data is inside a weird field called #36#37 that I have to dive into...
    r = res.callback_collocation.callback_collocation
    fields = fieldnames(typeof(r))
    r_deep = getproperty(r, fields[1])
        # This holds the contents of the #36#37 variable

    #r_deep.verbose
        # Things like this work

    # The TrainingResults object did not properly store the loss history in res.losses_collocation
    # Instead, you have to go to the #36#37 object


    time_scale = 10

    confirm_time_scale = true
    if confirm_time_scale

        println("")
        println("Confirming the assumed time_scale matches what was used... Check that the following two values are close:")
        
        # Internal time scaling that was used for training purposes
        println(r_deep.t_d[end] * time_scale / 60 )

        # Known data time
        println(convert(Float64, Dates.value(impedances[1].times[end] - impedances[1].times[1])) / (1000.0 * 60 * 60))

        println("If they are, then the time_scale of " * string(time_scale) * " is correct.")
        println("")

    end

    # The original data is not stored in the JLD2 object, I guess because I already had it stored in files.
    # For collocation, the training data is dd_smoothed, and r_deep.reference_data = res.callback_collocation.collocation_loss_choice.dd_smoothed

    # The data is internally scaled in the training algorithm.
    # This was copy-pasted, but we've gotta verify that it's correct.
    # We're going to do that by reconstructing the reference_data and making sure it matches the recorded reference_data
    scales = ( i->maximum(abs.(pca[i, :])) ).(1:size(pca, 1)) .* 1.25f0
    d = deepcopy(pca)
    d = d ./ scales
    t_d = deepcopy(timings[i])
    t_d .= t_d ./ time_scale

    d_smoothed = res.callback_collocation.collocation_loss_choice.d_smoothed
    dd_smoothed = res.callback_collocation.collocation_loss_choice.dd_smoothed

    # Smoothed data
    for k in 1:4
        if k == 1
            p = plot(timings[i] ./ 60, d_smoothed[k, :], color = k, label = "u" * subscriptnumber(k), xlabel = "Time (hours)", ylabel = "Normalized Data uᵢ (arb.)")
        else
            plot!(p, timings[i] ./ 60, d_smoothed[k, :], color = k, label = "u" * subscriptnumber(k))
        end
        plot!(p, timings[i] ./ 60, d[k, :], color = k, linealpha = 0.5, label = "")
    end
    p = plot(p, size = (600, 400), dpi = 800)#, labels = reshape(["u1"; "u2"; "u3"; "u4"; ""; ""; ""; ""], (1, 8)))
    display(p)
    savefig(p, "smoothed_data_" * string(impedances[i].name) * ".pdf")
    
    if false
        # Derivative of smoothed data vs. derivative of raw data
        p = plot(transpose(dd_smoothed))
        dd = finite_difference(d, t_d[2] - t_d[1])
        plot!(p, transpose(dd))
        display(p)
            # ...okay, yeah, that's why we do the derivative of smoothed data! This is a mess.
    end


    # Internal scaling - DO NOT USE (plot labels are wrong for this case)
    #plot_training_progress( r_deep.loss_trend, 
    #                        r_deep.t_d, # training_times, in minutes, scaled
    #                        r_deep.reference_data, # training_data, scaled
    #                        res.p_collocation, # best_parameters
    #                        NN_1 ) # model

    # True scale (for all appearance of u, t, and du/dt)
    plot_training_progress( r_deep.loss_trend, 
                            r_deep.t_d * time_scale / 60, # training_times, in hours
                            d_smoothed .* scales, # data, a.k.a. SMOOTHED u(t), for collocation (should be plain "pca" for SHOOTING, unsmoothed) 
                            r_deep.reference_data .* scales ./ time_scale * 60, # training_data, in Ohm / hour or Deg / hour
                            res.p_collocation, # best_parameters
                            NN_1, # model generator
                            time_scale, # time scale
                            scales, # u scale
                            name = imp.name, # For file saving
                            original_data = pca ) # ONLY for collocation

                                # NOTE: In the future, it probably shouldn't be d_smoothed for this data...probably should be the raw data, d.


    if i == 1
        # The L2 loss is weighted, in an effort to improve fits in COLLOCATION:
        s = size(dd_smoothed, 2)
        loss_weight = 1f0 .+ 20f0 * exp.(-(1:s)./80f0) # Loss weight 21 at start, decaying over a time scale of 80 samples down to 1 eventually
            # This is the formula that was used in the Fall 2020 data
        p = plot(t_d * time_scale / 60, loss_weight, xlabel = "Time (hours)", ylabel = "L2 Loss Weight (-)", legend = false, framestyle = :box, minorticks = true) # , title = "COLLOCATION ONLY - Loss weight"
        plot!(t_d * time_scale / 60, zeros(length(t_d)), linestyle = :dash, linecolor = :black)
        plot!(t_d * time_scale / 60, ones(length(t_d)), linestyle = :dash, linecolor = :black)
            # The shooting method does NOT use this same weight
            # Point is to make sure the derivative at the start is correct, so you don't go spiraling off in totally the wrong direction
        p = plot(p, size = (450, 300), dpi = 800)
        display(p)
        savefig(p, "COLLOCATION_ONLY_loss_weights.pdf")
    end

    verify_best = false
    if verify_best
        # Verify that p_collocation is (indeed) the *best* value
        println(res.p_collocation[1:10])
        println(res.p_history_collocation[argmin(r_deep.loss_trend)][1:10])
        println(res.p_collocation == res.p_history_collocation[argmin(r_deep.loss_trend)])
            # It is :)
    end


    ##### SHOOTING

    # Some of the data is inside a weird field called #36#37 that I have to dive into...
    r = res.callback_shooting.callback_shooting
    fields = fieldnames(typeof(r))
    r_deep = getproperty(r, fields[1])
        # This holds the contents of the #36#37 variable

    # True scale (for all appearance of u, t, and du/dt)
    plot_training_progress( r_deep.loss_trend, 
                            r_deep.t_d * time_scale / 60, # training_times, in hours
                            pca, # data (real-world units)
                            r_deep.reference_data .* scales, # training_data, in Ohm / hour or Deg / hour
                            res.θ_shooting, # best_parameters
                            NN_1, # model generator
                            time_scale, # time scale
                            scales, # u scale
                            name = imp.name, # For file saving
                            collocation = false)

                                # NOTE: In the future, it probably shouldn't be d_smoothed for this data...probably should be the raw data, d.



end
# Contains functions for visualization of data and training
# 
# Author: Harvey Merton

begin
    using Plots
    using Plots.PlotMeasures
end


#####
# Results visualization
#####

# Plot loss history during training
function plot_loss(train_data::TrainingData)
    plot(1:train_data.iter_cnt, train_data.L_hist, xlabel="Iteration", ylabel="Loss (MSE)", label=false, linewidth=2.5) #size=(400, 300) xlims=x_domain, seriestype = :scatter, legend = :topright, label = [string("$label_prefix$drone_ind", "_x") string("$label_prefix$drone_ind", "_y") string("$label_prefix$drone_ind", "_z")], xlabel = "Time (s)", ylabel = "$y_label", yformatter=)
end


# Plot data from a vector containing num_drones x 3-vectors over a time span given in t_data 
function plot_data!(plot_var, t_data, data::Vector{Vector{Vector{Float64}}}, plot_components::Bool, plot_legend::Bool, x_domain::Tuple{Float64, Float64}, label_prefix::String, y_label::String, seriestype, colors)   
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
        
        # Handle load case
        ind = "$drone_ind"
        if length(data[1]) == 1
            ind = "L"
        end

        if plot_components
            # Plot components for all time points seriestype =:scatter
            selected_colors = [colors[drone_ind][1] colors[drone_ind][2] colors[drone_ind][3]]
            
            if plot_legend
                plot!(plot_var, t_data, [comp1 comp2 comp3], xlims=x_domain, ylims=[-2.0, 2.0], seriestype = seriestype, linecolor = selected_colors, markercolor = selected_colors, label = [string("$label_prefix$ind", "_x") string("$label_prefix$ind", "_y") string("$label_prefix$ind", "_z")], xlabel = "Time (s)", ylabel = "$y_label", yformatter=two_dp_formatter) #legend = :topright,
            else
                if y_label == "Location (m)" #cmp(y_label, "Location (m)") == 0:
                    plot!(plot_var, t_data, [comp1 comp2 comp3], xlims=x_domain, ylims=[-1.0, 1.0], seriestype = seriestype, linecolor = selected_colors, markercolor = selected_colors, legend = false, xlabel = "Time (s)", ylabel = "$y_label", yformatter=two_dp_formatter)
                else
                    plot!(plot_var, t_data, [comp1 comp2 comp3], xlims=x_domain, ylims=[-5.0, 5.0], seriestype = seriestype, linecolor = selected_colors, markercolor = selected_colors, legend = false, xlabel = "Time (s)", ylabel = "$y_label", yformatter=two_dp_formatter)
                end
            end
        else
            # Plot magnitude for all time points
            selected_colors = colors[drone_ind][1]

            plot!(plot_var, t_data, magnitude, xlims=x_domain, seriestype = seriestype, linecolor = selected_colors, markercolor = selected_colors, legend = :topright, label = "$label_prefix$ind", xlabel = "Time (s)", ylabel = "$y_label", yformatter=two_dp_formatter) #right_margin = 5mm) #margin=(0mm, 5mm, 0mm, 0mm))
        end

    end

    return plot_var

end


# Convert data from vector of ArrayPartition form to nested vectors as plotting function was written assuming this data form
function convert_arrays_for_plot(data, t_data, accel_data, drone_swarm_params::DroneSwarmParams, is_load::Bool, is_angular::Bool, have_accel::Bool)
    # Convert array of ArrayPartitions into form that plotting function uses
    # Preallocate
    if is_load
        x_data = [[Vector{Float64}(undef, 3)] for _ in 1:length(t_data)]
        ẋ_data = [[Vector{Float64}(undef, 3)] for _ in 1:length(t_data)]
        ẍ_data = [[Vector{Float64}(undef, 3)] for _ in 1:length(t_data)]
    else
        x_data = [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] for _ in 1:length(t_data)] #for _ in drone_swarm_params.num_drones
        ẋ_data = [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] for _ in 1:length(t_data)]
        ẍ_data = [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] for _ in 1:length(t_data)]
    end

    # Pull out relevant data
    for i in 1:length(t_data)
        if (is_load && !is_angular) # Load linear 
            x_data[i] .= [data[i].x[1+4*drone_swarm_params.num_drones]]
            ẋ_data[i] .= [data[i].x[2+4*drone_swarm_params.num_drones]]

            if have_accel
                ẍ_data[i] .= [accel_data[i]]
            end
        elseif (is_load && is_angular) # Load angular 
            x_data[i] .= [data[i].x[3+4*drone_swarm_params.num_drones]]
            ẋ_data[i] .= [data[i].x[4+4*drone_swarm_params.num_drones]]

            if have_accel
                ẍ_data[i] .= [accel_data[i]]
            end
        elseif (!is_load && !is_angular) # Drone linear
            tmp = [data[i].x[j] for j in 1:drone_swarm_params.num_drones]
            x_data[i] = tmp

            tmp = [data[i].x[j] for j in drone_swarm_params.num_drones+1:2*drone_swarm_params.num_drones]
            ẋ_data[i] = tmp 
            
            if have_accel
                tmp = accel_data[i]
                ẍ_data[i] = tmp
            end
        else # Drone angular
            tmp = [data[i].x[j] for j in 2*drone_swarm_params.num_drones+1:3*drone_swarm_params.num_drones]
            x_data[i] = tmp

            tmp = [data[i].x[j] for j in 3*drone_swarm_params.num_drones+1:4*drone_swarm_params.num_drones]
            ẋ_data[i] = tmp
            
            if have_accel
                tmp = accel_data[i]
                ẍ_data[i] = tmp
            end
        end

    end

    return x_data, ẋ_data, ẍ_data

end


# Plot trajectory data generated for training the NN
# TODO: Make plotting work directly with ArrayPartition rather than having to convert twice (current plotting code temporary as change data type to ArrayPartition)
function plot_trajectory(t_data, data, accel_data, t_pred, u_pred, drone_swarm_params::DroneSwarmParams, is_load::Bool, is_angular::Bool, plot_sol::Bool) #x_data::Union{Vector{Vector{Float64}}, Vector{Vector{Vector{Float64}}}}, ẋ_data::Union{Vector{Vector{Float64}}, Vector{Vector{Vector{Float64}}}}, #ẍ_data::Union{Vector{Vector{Float64}}, Vector{Vector{Vector{Float64}}}}, is_load::Bool, is_angular::Bool)
    ## Pre-process data
    x_data, ẋ_data, ẍ_data = convert_arrays_for_plot(data, t_data, accel_data, drone_swarm_params, is_load, is_angular, true)

    ## Set up plots
    x_domain = (t_data[1], t_data[end]+((t_data[2] - t_data[1])/10))

    ## Set title
    if is_load
        plot_title = "Load Trajectory"
    else
        plot_title = "Drone Trajectories"
    end

    ## Set axes labels
    y_axis_label = "Location (m)"
    y_axis_label_dot = "Velocity (m/s)"
    y_axis_label_ddot = "Acceleration (m/s²)"

    legend_label = "x"
    legend_label_dot = "ẋ"
    legend_label_ddot = "ẍ"

    # Handle case for angular data
    if is_angular
        y_axis_label = "θ (rad)"
        y_axis_label_dot = "Ω (rad/s)"
        y_axis_label_ddot = "α (rad/s²)"

        legend_label = "θ"
        legend_label_dot = "Ω"
        legend_label_ddot = "α"

    end

    ## Plot trajectory data and prediction
    seriestype_data = :scatter
    colors = [[:green, :blue, :purple], [:red, :lime, :magenta], [:black, :grey, :aqua]] #x, y, z components for all drones (load will simply take first set)

    # Position
    p_x = plot()
    plot_data!(p_x, t_data, x_data, true, false, x_domain, legend_label, y_axis_label, seriestype_data, colors) # data

    # Velocity
    p_ẋ = plot()
    plot_data!(p_ẋ, t_data, ẋ_data, true, true, x_domain, legend_label_dot, y_axis_label_dot, seriestype_data, colors)

    # Acceleration
    p_ẍ = plot()
    plot_data!(p_ẍ, t_data, ẍ_data, true, false, x_domain, legend_label_ddot, y_axis_label_ddot, seriestype_data, colors)
    
    # Plot predicted solution alongside data if requested (no acceleration prediciton as acceleration not in state)
    if plot_sol
        ## Convert 
        x_pred, ẋ_pred, _ = convert_arrays_for_plot(u_pred, t_pred, accel_data, drone_swarm_params, is_load, is_angular, false)

        ## Plot
        seriestype_pred = :line

        # Position prediction
        plot_data!(p_x, t_pred, x_pred, true, false, x_domain, legend_label, y_axis_label, seriestype_pred, colors)

        # Velocity prediction
        plot_data!(p_ẋ, t_pred, ẋ_pred, true, true, x_domain, legend_label_dot, y_axis_label_dot, seriestype_pred, colors)


    end

    ## Display trajectory
    p_traj = plot(p_x, p_ẋ, p_ẍ, layout=(3,1), size=(800, 600)) #, title=plot_title)
    display(p_traj)
end

# Plot trajectory data vs prediction from solving ODE with tension approximated by NN
function plot_pred_vs_data(t_data_plot, data_plot, ẍₗ_trimmed, αₗ_trimmed, ẍᵢ_trimmed, αᵢ_trimmed, p_nn, drone_swarm_params, u0, data, ẍₗ, αₗ, ẍᵢ, αᵢ, t_data, step_first)
    # Solve system
    t_span = (t_data_plot[1], t_data_plot[end]) 
    time_save_points = t_span[1]:(t_data_plot[2]-t_data_plot[1]):t_span[2] # Assuming data saved at fixed-distance points round(, digits=3)
    sol = solve_ode_system(drone_swarm_params, time_save_points, u0, p_nn, false, 0.1, data, ẍₗ, αₗ, ẍᵢ, t_data, step_first)

    # Load
    plot_trajectory(t_data_plot, data_plot, ẍₗ_trimmed, sol.t, sol.u, drone_swarm_params, true, false, true) # Linear
    plot_trajectory(t_data_plot, data_plot, αₗ_trimmed, sol.t, sol.u, drone_swarm_params, true, true, true) # Angular

    # Drone
    plot_trajectory(t_data_plot, data_plot, ẍᵢ_trimmed, sol.t, sol.u, drone_swarm_params, false, false, true) # Linear
    plot_trajectory(t_data_plot, data_plot, αᵢ_trimmed, sol.t, sol.u, drone_swarm_params, false, true, true) # Angular

end

# Plot data generated for training the NN
function plot_tension_nn_ip_op(t_data, T_data::Vector{Vector{Vector{Float64}}}, x₍i_rel_Lᵢ₎::Vector{Vector{Vector{Float64}}}, ẋ₍i_rel_Lᵢ₎::Vector{Vector{Vector{Float64}}}, ẍ₍i_rel_Lᵢ₎::Vector{Vector{Vector{Float64}}}, plot_components::Bool, plot_traj::Bool, plot_pred::Bool, drone_swarm_params, p_nn_T_drone)
    x_domain = (t_data[1], t_data[end]+((t_data[2] - t_data[1])/10))
    
    ## Plot tension data
    seriestype_data = :scatter
    colors = [[:green, :blue, :purple], [:red, :lime, :magenta], [:black, :grey, :aqua]]

    p_tension = plot()
    plot_data!(p_tension, t_data, T_data, plot_components, true, x_domain, "T_data_","Tension (N)", seriestype_data, colors)
    

    ## Plot trajectory data
    # Position
    p_x = plot()
    plot_data!(p_x, t_data, x₍i_rel_Lᵢ₎, plot_components, false, x_domain, "x₍i_rel_Lᵢ₎_data_", "Location (m)", seriestype_data, colors)

    # Velocity
    p_ẋ = plot()
    plot_data!(p_ẋ, t_data, ẋ₍i_rel_Lᵢ₎, plot_components, true, x_domain, "ẋ₍i_rel_Lᵢ₎_data_", "Velocity (m/s)", seriestype_data, colors) #t_data[2:end]

    # Acceleration
    p_ẍ = plot()
    plot_data!(p_ẍ, t_data, ẍ₍i_rel_Lᵢ₎, plot_components, false, x_domain, "ẍ₍i_rel_Lᵢ₎_data_", "Acceleration (m/s²)", seriestype_data, colors) #t_data[3:end]


    ## Plot trajectory data
    seriestype_data = :line
    # Position
    plot_data!(p_x, t_data, drone_swarm_params.x₍i_rel_Lᵢ₎_hist, plot_components, false, x_domain, "x₍i_rel_Lᵢ₎_", "Location (m)", seriestype_data, colors)

    # Velocity
    plot_data!(p_ẋ, t_data, drone_swarm_params.ẋ₍i_rel_Lᵢ₎_hist, plot_components, true, x_domain, "ẋ₍i_rel_Lᵢ₎_", "Velocity (m/s)", seriestype_data, colors) 

    # Acceleration
    plot_data!(p_ẍ, t_data, drone_swarm_params.ẍ₍i_rel_Lᵢ₎_hist, plot_components, false, x_domain, "ẍ₍i_rel_Lᵢ₎_", "Acceleration (m/s²)", seriestype_data, colors) 



    ## Plot tension predicitons if requested
    T_pred = [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] for _ in 1:length(t_data)]


    if plot_pred
        seriestype_pred = :line
        nn_T_drone = drone_swarm_params.re_nn_T_drone(p_nn_T_drone)

        # Get prediction of cable tension vector at every timestep for every cable
        for step_num in eachindex(T_data)
            for cable_num in eachindex(T_data[1])
                nn_ip = vcat(x₍i_rel_Lᵢ₎[step_num][cable_num], ẋ₍i_rel_Lᵢ₎[step_num][cable_num], ẍ₍i_rel_Lᵢ₎[step_num][cable_num])
                nn_ip = convert.(Float32, nn_ip)
                
                T_pred[step_num][cable_num] = nn_T_drone(nn_ip)
            end
        end

        plot_data!(p_tension, t_data, T_pred, plot_components, true, x_domain, "T_pred_","Tension (N)", seriestype_pred, colors)
    end


    ## Display plots
    # Tensions
    display(p_tension)

    if plot_traj
        # Trajectory
        p_traj = plot(p_x, p_ẋ, p_ẍ, layout=(3,1), size=(800, 600))
        display(p_traj)
    end

end

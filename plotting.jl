# Contains functions for visualization of data and training
# 
# Author: Harvey Merton

begin
    # using Plots
    # using Plots.PlotMeasures
end


#####
# Results visualization
#####

# Plot loss history during training
function plot_loss(train_data::TrainingData)
    plot(1:train_data.iter_cnt, train_data.L_hist, xlabel="Iteration", ylabel="Loss (MSE)", label=false, linewidth=2.5) #size=(400, 300) xlims=x_domain, seriestype = :scatter, legend = :topright, label = [string("$label_prefix$drone_ind", "_x") string("$label_prefix$drone_ind", "_y") string("$label_prefix$drone_ind", "_z")], xlabel = "Time (s)", ylabel = "$y_label", yformatter=)
end


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
            plot!(plot_var, t_data, [comp1, comp2, comp3], xlims=x_domain, seriestype = :scatter, legend = :topright, label = [string("$label_prefix$drone_ind", "_x") string("$label_prefix$drone_ind", "_y") string("$label_prefix$drone_ind", "_z")], xlabel = "Time (s)", ylabel = "$y_label", yformatter=two_dp_formatter)
        else
            # Plot magnitude for all time points
            plot!(plot_var, t_data, magnitude, xlims=x_domain, seriestype = :scatter, legend = :topright, label = "$label_prefix$drone_ind", xlabel = "Time (s)", ylabel = "$y_label", yformatter=two_dp_formatter) #right_margin = 5mm) #margin=(0mm, 5mm, 0mm, 0mm))
        end

    end

    return plot_var

end


# Plot trajectory data generated for training the NN
# TODO: Make plotting work directly with ArrayPartition rather than having to convert twice (current plotting code temporary as change data type to ArrayPartition)
function plot_trajectory(t_data, data, accel_data, drone_swarm_params::DroneSwarmParams, p_nn_T_drone, is_load::Bool, is_angular::Bool) #x_data::Union{Vector{Vector{Float64}}, Vector{Vector{Vector{Float64}}}}, ẋ_data::Union{Vector{Vector{Float64}}, Vector{Vector{Vector{Float64}}}}, #ẍ_data::Union{Vector{Vector{Float64}}, Vector{Vector{Vector{Float64}}}}, is_load::Bool, is_angular::Bool)
    ## Pre-process data
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
            ẍ_data[i] .= [accel_data[i]]
        elseif (is_load && is_angular) # Load angular 
            x_data[i] .= [data[i].x[3+4*drone_swarm_params.num_drones]]
            #println(x_data[i])
            ẋ_data[i] .= [data[i].x[4+4*drone_swarm_params.num_drones]]
            ẍ_data[i] .= [accel_data[i]]
        elseif (!is_load && !is_angular) # Drone linear
            tmp = [data[i].x[j] for j in 1:drone_swarm_params.num_drones]
            x_data[i] = tmp

            tmp = [data[i].x[j] for j in drone_swarm_params.num_drones+1:2*drone_swarm_params.num_drones]
            ẋ_data[i] = tmp 
            
            tmp = accel_data[i]
            ẍ_data[i] = tmp
        else # Drone angular
            tmp = [data[i].x[j] for j in 2*drone_swarm_params.num_drones+1:3*drone_swarm_params.num_drones]
            x_data[i] = tmp

            tmp = [data[i].x[j] for j in 3*drone_swarm_params.num_drones+1:4*drone_swarm_params.num_drones]
            ẋ_data[i] = tmp
            
            tmp = accel_data[i]
            ẍ_data[i] = tmp
        end

    end


    

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
    # Position
    p_x = plot()
    plot_data!(p_x, t_data, x_data, true, x_domain, legend_label, y_axis_label) # data

    # Velocity
    p_ẋ = plot()
    plot_data!(p_ẋ, t_data, ẋ_data, true, x_domain, legend_label_dot, y_axis_label_dot)

    # Acceleration
    p_ẍ = plot()
    plot_data!(p_ẍ, t_data, ẍ_data, true, x_domain, legend_label_ddot, y_axis_label_ddot)
    
    ## Display trajectory
    p_traj = plot(p_x, p_ẋ, p_ẍ, layout=(3,1), size=(800, 600), title=plot_title)
    display(p_traj)
end


# Plot data generated for training the NN
function plot_tension_nn_ip_op(t_data, T_data::Vector{Vector{Vector{Float64}}}, x₍i_rel_Lᵢ₎::Vector{Vector{Vector{Float64}}}, ẋ₍i_rel_Lᵢ₎::Vector{Vector{Vector{Float64}}}, ẍ₍i_rel_Lᵢ₎::Vector{Vector{Vector{Float64}}}, plot_components::Bool)
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






# # Plot trajectory data generated for training the NN
# function plot_trajectory_old(t_data, x::Union{Vector{Vector{Float64}}, Vector{Vector{Vector{Float64}}}}, ẋ::Union{Vector{Vector{Float64}}, Vector{Vector{Vector{Float64}}}}, ẍ::Union{Vector{Vector{Float64}}, Vector{Vector{Vector{Float64}}}}, is_load::Bool, is_angular::Bool)
#     x_domain = (t_data[1], t_data[end]+((t_data[2] - t_data[1])/10))
#     plot_title = "Drone Trajectories"
    
#     ## Hanle case for load
#     if is_load
#         plot_title = "Load Trajectory"

#         # Repackage load data to work with plot function
#         x_new = [[Vector{Float64}(undef, 3)] for _ in 1:length(t_data)]
#         ẋ_new = [[Vector{Float64}(undef, 3)] for _ in 1:length(t_data)]
#         ẍ_new = [[Vector{Float64}(undef, 3)] for _ in 1:length(t_data)]

#         for i in 1:length(t_data)
#             x_new[i] = [x[i]]
#             ẋ_new[i] = [ẋ[i]]
#             ẍ_new[i] = [ẍ[i]]
#         end
#     else
#         x_new = x
#         ẋ_new = ẋ
#         ẍ_new = ẍ
#     end

#     ## Set axes labels
#     y_axis_label = "Location (m)"
#     y_axis_label_dot = "Velocity (m/s)"
#     y_axis_label_ddot = "Acceleration (m/s²)"

#     legend_label = "x"
#     legend_label_dot = "ẋ"
#     legend_label_ddot = "ẍ"

#     # Handle case for angular data
#     if is_angular
#         y_axis_label = "θ (rad)"
#         y_axis_label_dot = "Ω (rad/s)"
#         y_axis_label_ddot = "α (rad/s²)"

#         legend_label = "θ"
#         legend_label_dot = "Ω"
#         legend_label_ddot = "α"

#     end

#     ## Plot trajectory data
#     # Position
#     p_x = plot()
#     plot_data!(p_x, t_data, x_new, true, x_domain, legend_label, y_axis_label)

#     # Velocity
#     p_ẋ = plot()
#     plot_data!(p_ẋ, t_data, ẋ_new, true, x_domain, legend_label_dot, y_axis_label_dot)

#     # Acceleration
#     p_ẍ = plot()
#     plot_data!(p_ẍ, t_data, ẍ_new, true, x_domain, legend_label_ddot, y_axis_label_ddot)
    
#     ## Display trajectory
#     p_traj = plot(p_x, p_ẋ, p_ẍ, layout=(3,1), size=(800, 600), title=plot_title)
#     display(p_traj)
# end

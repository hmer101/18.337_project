# Helper functions for solving ODE system and training NN
# 
# Author: Harvey Merton

begin
    # include("utils.jl")
    # include("datatypes.jl")
end

##########
# SOLVING
##########

# Solve the drone swarm ODE system
function solve_ode_system(drone_swarm_params, time_save_points, u0, p_nn_T_drone, adaptive_true, solve_dt, data, data_ẍₗ, data_αₗ, data_ẍᵢ, t_data, step_first)
    # Reset cache before solving ODE
    reset_cache!(drone_swarm_params, t_data, data, data_ẍₗ, data_αₗ, data_ẍᵢ, step_first)
    
    # Solve the ODE
    prob = ODEProblem(drone_swarm_params, u0, (time_save_points[1], time_save_points[end]), p_nn_T_drone)
    sol = solve(prob, Tsit5(), saveat=time_save_points, adaptive=adaptive_true, dt=solve_dt)

    # Reset cache after solving ODE so non-mutating
    #reset_cache!(drone_swarm_params, t_data, data, data_ẍₗ, data_αₗ, data_ẍᵢ, step_first)

    return sol
end

# Reset the cache in drone_swarm_params
# Note that drone_swarm_params needs to be initialized first
# Note that original data and t_data put in, not trimmed data
function reset_cache!(drone_swarm_params, t_data, data, ẍₗ, αₗ, ẍᵢ, step_first)
    # Only need to reset cache values (_prev)
    # Don't need to change data or constants are these are unchanged when running
    drone_swarm_params.t_prev = deepcopy(t_data[step_first-1])

    # Velocities (TODO: DELETE AS NO LONGER USED)
    num_drones = drone_swarm_params.num_drones
    drone_swarm_params.ẋₗ_prev = deepcopy(data[step_first-1].x[2+4*num_drones])
    drone_swarm_params.Ωₗ_prev = deepcopy(data[step_first-1].x[4+4*num_drones])
    drone_swarm_params.ẋᵢ_prev = deepcopy(collect(data[step_first-1].x[num_drones+1:2*num_drones]))

    # Accelerations (TODO: Only need first value, not entire array, passed in)
    drone_swarm_params.ẍₗ_prev = deepcopy(ẍₗ[step_first-1]) 
    drone_swarm_params.αₗ_prev = deepcopy(αₗ[step_first-1]) 
    drone_swarm_params.ẍᵢ_prev = deepcopy(ẍᵢ[step_first-1])

end


##########
# TRAINING
##########

# Take data and t_data so can reset cache in drone_swarm_params therefore not mutating
# TODO: Group loss params into struct
function loss(data_trimmed, t_data_trimmed, data, data_ẍₗ, data_αₗ, data_ẍᵢ, t_data, drone_swarm_params, u0, p_nn_T_drone, step_first)
    # Solve 
    #TODO: Turn adaptive solve back on and set dt properly
    t_span = (t_data_trimmed[1], t_data_trimmed[end]) 
    time_save_points = t_span[1]:(t_data_trimmed[2]-t_data_trimmed[1]):t_span[2] # Assuming data saved at fixed-distance points round(, digits=3)
    sol = solve_ode_system(drone_swarm_params, time_save_points, u0, p_nn_T_drone, false, 0.1, data, data_ẍₗ, data_αₗ, data_ẍᵢ, t_data, step_first)
    
    # Get loss relative to data
    l = 0
    for step_num in 1:length(t_data_trimmed) #size(data,2)
        l += mean((data_trimmed[step_num] - sol.u[step_num]).^2) #sum rather than mean??
    end

    return l
end

# Callback function during training to store and print loss information
function (train_data::TrainingData)(p_nn_T_drone, l_current) #; doplot = false) 
    # Print iteration and loss 
    iter_temp = train_data.iter_cnt
    print("Iter: $iter_temp ")
    println("Loss: $l_current")

    # if doplot
    #     pl, _ = visualize(p_nn_T_drone)
    #     display(pl)
    # end

    #push!(train_data.L_hist, l_current) # For non pre-allocated L_hist
    train_data.iter_cnt += 1
    train_data.L_hist[train_data.iter_cnt] = l_current

    return false
end
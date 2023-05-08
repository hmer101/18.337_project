# 18.337 Project - Learning cable tension vectors in multi-drone slung load carrying
# 
# Author: Harvey Merton


begin
   using DifferentialEquations 
   using Flux
   using DiffEqFlux
   #using DiffEqCallbacks
   
   #using Optimization
   using OptimizationOptimisers
   using FiniteDiff
   using FiniteDifferences

#    using ForwardDiff
#    using ReverseDiff
#    using Zygote 
#    using Tracker

   using Rotations
   using LinearAlgebra
   using Statistics

   using RecursiveArrayTools
   
   using Plots
   using Plots.PlotMeasures

   import AbstractDifferentiation as AD

   include("datatypes.jl")
   include("utils.jl")
   include("plotting.jl")
   include("generateData.jl")
   include("solveTrain.jl")
end

begin
    const NUM_DRONES = 3 
end

# Function that returns num_drones drone inputs (forces and moments) at a specified time
# Currently set values close to quasi-static state-regulation inputs. TODO: might need to change to include horizonal components 
# TODO: Perhaps try trajectory-following
function drone_forces_and_moments(params, t::Float64)
    # This uses a reasonable estimate of fₘ for the static trajectory
    # # Preallocate the output vector
    # fₘ = [[0.0, Vector{Float64}(undef, 3)] for i in 1:params.num_drones]
    
    # # Store the force and moment inputs for each drone
    # for i in 1:params.num_drones
    #     fₘ[i][1] = (params.m_drones[i] + params.m_cables[i] + params.m_load/params.num_drones)*params.g # Force
    #     fₘ[i][2] = [0.0, 0.0, 0.0] # Moment
    # end

    # This uses the forces and moments calculated in the data required to make the swarm follow the desired trajectory
    # Do a "shifted zero-order-hold" on input data
    f_ind = convert(Int, round((t-params.t_start)/params.t_step)) + 1

    f_m = params.fₘ[f_ind]

    #TODO: Throw error if using untrimmed data

    return f_m
end

# u = x = []
# TODO: replace all du= with du.= (rather than [1:end])?? Make sure to copy where required
function (params::DroneSwarmParams)(du,u,p,t) #ode_sys_drone_swarm_nn!(du,u,p,t)
    # Get force and moment inputs from drones
    fₘ = drone_forces_and_moments(params, t)      # TODO: FIX THIS -> how coming through in training data??? Output force required in training data then generate this force for each drone in the generate_forces_and_moments function   Perhaps set a force in training data rather than desired load trajectory?? Or simply use desired load trajectory formulation here w/o force

    ## Variable unpacking 
    e₃ = [0,0,1] 

    # Reconstruct NN for tension estimation
    nn_T_drone = params.re_nn_T_drone(p)

    # Load
    xₗ = u.x[params.num_drones*4+1]
    ẋₗ = u.x[params.num_drones*4+2]
    θₗ = u.x[params.num_drones*4+3]
    Ωₗ = u.x[params.num_drones*4+4]

    Rₗ = rpy_to_R([θₗ[1], θₗ[2], θₗ[3]]) # RPY angles to rotation matrix

    ## Equations of motion
    # Accumulate effect of tension on load from all drones 
    ∑RₗTᵢ_load = zeros(3)
    ∑rᵢxTᵢ_load = zeros(3)

    # Load acceleration estimations
    # Rather than using backwards finite difference, simply estimate current acceleration as previous actual acceleration
    # Backwards finite difference has lag anyway, this just avoids numerical error amplification by small timesteps
    ẍₗ_est = params.ẍₗ_prev
    αₗ_est = params.αₗ_prev


    # All drones
    for i in 1:params.num_drones
        ### Variable unpacking
        # Drone states
        xᵢ = u.x[i]
        ẋᵢ = u.x[params.num_drones+i]
        θᵢ = u.x[2*params.num_drones+i]
        Ωᵢ = u.x[3*params.num_drones+i] # Same as θ̇ᵢ

        Rᵢ = rpy_to_R([θᵢ[1], θᵢ[2], θᵢ[3]]) # RPY angles to rotation matrix. 

        # Inputs 
        fᵢ = fₘ[i][1]
        mᵢ = fₘ[i][2]

        ### Calculate tension
        Lᵢqᵢ = -inv(Rₗ)*(xᵢ - xₗ) + params.r_cables[i]

        ## Drone relative to attachment point
        # Position
        r₍i_rel_Lᵢ₎ = -Lᵢqᵢ
        x₍i_rel_Lᵢ₎ = r₍i_rel_Lᵢ₎
        
        # Velocity
        ẋ₍i_rel_Lᵢ₎ = ẋᵢ - (ẋₗ + cross(Ωₗ, params.r_cables[i]))

        # Acceleration
        ẍᵢ_est = params.ẍᵢ_prev[i] # Estimate drone acceleration as previous timestep acceleration (using backwards FD in velocities exacerbates error)

        ẍ₍Lᵢ₎ = ẍₗ_est + cross(αₗ_est, params.r_cables[i]) + cross(Ωₗ, cross(Ωₗ, params.r_cables[i]))
        ẍ₍i_rel_Lᵢ₎ = ẍᵢ_est - ẍ₍Lᵢ₎ - cross(αₗ_est, r₍i_rel_Lᵢ₎) - cross(Ωₗ, cross(Ωₗ,r₍i_rel_Lᵢ₎)) - 2*cross(Ωₗ,ẋ₍i_rel_Lᵢ₎)
 

        ## Drone side
        nn_ip = vcat(x₍i_rel_Lᵢ₎, ẋ₍i_rel_Lᵢ₎, ẍ₍i_rel_Lᵢ₎)
        nn_ip = convert.(Float32, nn_ip)

        Tᵢ_drone = nn_T_drone(nn_ip)
        #Tᵢ_load = -Tᵢ_drone # TODO: Note this Ti_load = -Ti_drone relationship will not hold without assumption

        # CAN CHECK TENSION VECTOR direction
        # qᵢ = T₍ind₎[i]/norm(T₍ind₎[i])
        # Lᵢqᵢ = params.l_cables[i]*qᵢ

        # Sum across all cables needed for load EOM calculations - Might not use Rl when not using assumption??
        ∑RₗTᵢ_load += Rₗ*Tᵢ_drone # Forces
        ∑rᵢxTᵢ_load += cross(params.r_cables[i],-Tᵢ_drone) # Moments

        
        ### Equations of motion
        ## Drones
        # Velocity
        du.x[i][1:length(du.x[i])] = ẋᵢ[1:end]

        # Acceleration
        ẍᵢ = (1/params.m_drones[i])*(fᵢ*Rᵢ*e₃ - params.m_drones[i]*params.g*e₃ + Rₗ*Tᵢ_drone) # Might not use Rl when not using assumption?????
        du.x[i+params.num_drones][1:length(du.x[i+params.num_drones])] = ẍᵢ[1:end]

        # Angular velocity
        du.x[i+2*params.num_drones][1:length(du.x[i+2*params.num_drones])] = Ωᵢ[1:end]

        # Angular acceleration
        # αᵢ = inv(p.j_drones[i])*(mᵢ - cross(Ωᵢ,(p.j_drones[i]*Ωᵢ)))
        du.x[i+3*params.num_drones][1:length(du.x[i+3*params.num_drones])] = inv(params.j_drones[i])*(mᵢ - cross(Ωᵢ,(params.j_drones[i]*Ωᵢ)))
        

        ### Update cache
        params.ẋᵢ_prev[i][1:end] = deepcopy(ẋᵢ[1:end])
        params.ẍᵢ_prev[i][1:end] = deepcopy(ẍᵢ[1:end]) #_est

    end

    ## Load EOM
    # Velocity
    du.x[1+4*params.num_drones][1:length(du.x[1+4*params.num_drones])] = ẋₗ[1:end]

    # Acceleration
    ẍₗ = (1/params.m_load)*(-∑RₗTᵢ_load-params.m_load*params.g*e₃)
    du.x[2+4*params.num_drones][1:length(du.x[2+4*params.num_drones])] = ẍₗ[1:end]

    # Angular velocity
    du.x[3+4*params.num_drones][1:length(du.x[3+4*params.num_drones])] = Ωₗ[1:end]

    # Angular acceleration
    αₗ = inv(params.j_load)*(∑rᵢxTᵢ_load - cross(Ωₗ,(params.j_load*Ωₗ)))
    du.x[4+4*params.num_drones][1:length(du.x[4+4*params.num_drones])] = αₗ[1:end]

    ### Update cache
    params.t_prev = t

    params.ẋₗ_prev[1:end] = deepcopy(ẋₗ[1:end])
    params.Ωₗ_prev[1:end] = deepcopy(Ωₗ[1:end])
    
    params.ẍₗ_prev[1:end] = deepcopy(ẍₗ[1:end]) #_est
    params.αₗ_prev[1:end] = deepcopy(αₗ[1:end]) #_est

end



# GENERATE TRAINING DATA
begin
    t_step_data = 0.1
    j_drone = [2.32 0 0; 0 2.32 0; 0 0 4]

    params_training = DroneSwarmParams_init(num_drones=NUM_DRONES, g=9.81, m_load=0.225, m_drones=[0.5, 0.5, 0.5], m_cables=[0.1, 0.1, 0.1], l_cables=[1.0, 1.0, 1.0],
                                    j_load = [2.1 0 0; 0 1.87 0; 0 0 3.97], j_drones= [j_drone, j_drone, j_drone], 
                                    r_cables = [[-0.42, -0.27, 0], [0.48, -0.27, 0], [-0.06, 0.55, 0]], t_step=0.1, t_start=0.0, fₘ=[[[]]], re_nn_T_drone=nothing, 
                                    t_prev=0.0, ẋₗ_prev=ones(3), Ωₗ_prev=ones(3), ẋᵢ_prev=[ones(3), ones(3), ones(3)], 
                                    ẍₗ_prev=[0.0, 0.0, 0.0], αₗ_prev=[0.0, 0.0, 0.0], ẍᵢ_prev=Vector{Vector{Float64}}([zeros(3) for _ in 1:NUM_DRONES]))

    t_data, data, ẍᵢ, αᵢ, ẍₗ, αₗ, fₘ = generate_training_data(t_step_data, params_training) #t_data, xᵢ, ẋᵢ, ẍᵢ, θᵢ, Ωᵢ, αᵢ, xₗ, ẋₗ, ẍₗ, θₗ, Ωₗ, αₗ
    println()
end

# TRIM TRAINING DATA
begin
    # Trim training data to remove early numerical estimation errors
    step_first = 5 #4 should be enough as accel stable here and velocity stable at 2. Only need to reach back for veloctiy
    step_last = step_first+3 # TODO: REMOVE END TRIMMING (set step_last = end)
    
    t_data_trimmed = t_data[step_first:end] #step_last
    data_trimmed = data[step_first:end]

    ẍᵢ_trimmed = ẍᵢ[step_first:end]
    αᵢ_trimmed = αᵢ[step_first:end]
    ẍₗ_trimmed = ẍₗ[step_first:end]
    αₗ_trimmed = αₗ[step_first:end]

    fₘ_trimmed = fₘ[step_first:end]

end

# PLOT TRAINING DATA
begin
    # Select trimmed or untrimmed data
    t_data_plot = t_data_trimmed #t_data_trimmed #t_data
    data_plot = data_trimmed #data_trimmed #data

    # Load
    plot_trajectory(t_data_plot, data_plot, ẍₗ_trimmed, false, false, drone_swarm_params, true, false, false) # Linear
    plot_trajectory(t_data_plot, data_plot, αₗ_trimmed, false, false, drone_swarm_params, true, true, false) # Angular

    # Drone
    plot_trajectory(t_data_plot, data_plot, ẍᵢ_trimmed, false, false, drone_swarm_params, false, false, false) # Linear
    #plot_trajectory(t_data_plot[3:end], data_plot[3:end], ẍᵢ[3:end], drone_swarm_params, p_nn_T_drone, false, false) # Linear
    plot_trajectory(t_data_plot, data_plot, αᵢ_trimmed, false, false, drone_swarm_params, false, true, false) # Angular
end


# SET UP IC's and PARAMS
begin
    ## Set initial conditions
    u0 = similar(data_trimmed[1])
    
    # Add drones ICs
    for i in 1:NUM_DRONES
        ## Drones
        # Position
        u0.x[i] .= data_trimmed[1].x[i]

        # Velocity
        u0.x[NUM_DRONES+i] .= data_trimmed[1].x[NUM_DRONES+i]

        # Orientation
        u0.x[2*NUM_DRONES+i] .= data_trimmed[1].x[2*NUM_DRONES+i]

        # Angular velocity
        u0.x[3*NUM_DRONES+i] .= data_trimmed[1].x[3*NUM_DRONES+i]
        

    end
    # Load ICs
    # Position
    u0.x[1+4*NUM_DRONES] .= data_trimmed[1].x[1+4*NUM_DRONES]

    # Velocity
    u0.x[2+4*NUM_DRONES] .= data_trimmed[1].x[2+4*NUM_DRONES]

    # Orientation
    u0.x[3+4*NUM_DRONES] .= data_trimmed[1].x[3+4*NUM_DRONES]

    # Angular velocity
    u0.x[4+4*NUM_DRONES] .= data_trimmed[1].x[4+4*NUM_DRONES]

    ## Setup parameters
    # Cable tension NNs - take in flattened vector inputs, output tension vector at drone and load respectively
    input_dim = 9 # Position, velocity and acceleration vectors for each drone relative to the attachment point of their attached cables on the load
    T_drone_nn = Chain(Dense(input_dim, 20, tanh), Dense(20, 3)) #Chain(Dense(input_dim, 3, tanh)) #Chain(Dense(input_dim, 32, tanh), Dense(32, 3)) # TODO: Could try 2 hidden layers!!

    # Initialise parameter struct
    # Note that cache values are initialised at corresponding u0 values
    j_drone = [2.32 0 0; 0 2.32 0; 0 0 4]
    p_nn_T_drone, re_nn_T_drone = Flux.destructure(T_drone_nn)
    p_nn_T_drone_pre_train = copy(p_nn_T_drone) # Store pre-trained values for visualization

    drone_swarm_params = DroneSwarmParams_init(num_drones=NUM_DRONES, g=9.81, m_load=0.225, m_drones=[0.5, 0.5, 0.5], m_cables=[0.1, 0.1, 0.1], l_cables=[1.0, 1.0, 1.0],
                                    j_load = [2.1 0 0; 0 1.87 0; 0 0 3.97], j_drones= [j_drone, j_drone, j_drone], 
                                    r_cables = [[-0.42, -0.27, 0], [0.48, -0.27, 0], [-0.06, 0.55, 0]], t_step=t_step_data, t_start=t_data_trimmed[1], fₘ=fₘ_trimmed, re_nn_T_drone=re_nn_T_drone, 
                                    t_prev=t_data[step_first-1], ẋₗ_prev=data[step_first-1].x[2+4*NUM_DRONES], Ωₗ_prev=data[step_first-1].x[4+4*NUM_DRONES], ẋᵢ_prev=collect(data[step_first-1].x[NUM_DRONES+1:2*NUM_DRONES]), 
                                    ẍₗ_prev=ẍₗ[step_first-1], αₗ_prev=αₗ[step_first-1], ẍᵢ_prev=ẍᵢ[step_first-1])
    
    # Reset cache in struct
    #reset_cache!(drone_swarm_params, t_data, data, ẍₗ, αₗ, ẍᵢ, step_first)
end




# TEST: ONE ODE CALL
begin
    # du_temp = [Vector{Float64}(undef, 3) for i in 1:(4*drone_swarm_params.num_drones+4)]
    # du = ArrayPartition(du_temp[1],du_temp[2], du_temp[3], du_temp[4], 
    #     du_temp[5], du_temp[6], du_temp[7], du_temp[8], du_temp[9], du_temp[10],du_temp[11], du_temp[12], 
    #     du_temp[13],du_temp[14], du_temp[15], du_temp[16]) # Must be a better way of initializing. This doesn't work anyway -> probably need to do du_temp[a][1:length(du_temp[a])] trick
    reset_cache!(drone_swarm_params, t_data, data, ẍₗ, αₗ, ẍᵢ, step_first)

    du = ArrayPartition([2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], 
        [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], 
        [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]) 

    t = t_data_trimmed[1] #t_data[step_u0] # Test solve at initial datapoint

    # ## Test one function call
    #println("before: $u0") #$du")
    drone_swarm_params(du, u0, p_nn_T_drone, t) # params_sense
    # println()
    # println("du after: $du") #$du")
end


# TEST: DOES LOSS FXN CHANGE WITH CHANGES IN NN_PARMS?
begin
    # Run ODE solver (define better u0's, proper fi's and mi's !)
    #step_size_callback = DiscreteCallback(true, print_step_size) # TODO: Tune this (when save etc.)

    p_nn_T_drone_new = deepcopy(p_nn_T_drone)
    p_nn_T_drone_new[1:5] .= 5000.0
end

begin
    # Get baseline
    #reset_cache!(drone_swarm_params, t_data, data, ẍₗ, αₗ, ẍᵢ, step_first)

    println("l1")
    #println(drone_swarm_params)
    l1 = loss(data_trimmed, t_data_trimmed, data, ẍₗ, αₗ, ẍᵢ, t_data, drone_swarm_params, u0, p_nn_T_drone, step_first) #sol1
    println(l1) 

    # Compare
    #reset_cache!(drone_swarm_params, t_data, data, ẍₗ, αₗ, ẍᵢ, step_first)

    println()
    println("l2")
    #println(drone_swarm_params)
    l2 = loss(data_trimmed, t_data_trimmed, data, ẍₗ, αₗ, ẍᵢ, t_data, drone_swarm_params, u0, p_nn_T_drone_new, step_first) #sol2  
    println(l2) # Loss should be different to first
end


# TEST: Gradient
begin
    # Method 1 - FiniteDiff
    reset_cache!(drone_swarm_params, t_data, data, ẍₗ, αₗ, ẍᵢ, step_first)
    grad = FiniteDiff.finite_difference_gradient(p_nn_T_drone-> loss(data_trimmed, t_data_trimmed, data, ẍₗ, αₗ, ẍᵢ, t_data, drone_swarm_params, u0, p_nn_T_drone, step_first), p_nn_T_drone) # Easier
    
    # 1a - ensure reset cache working. Yes! Get same result as 1
    # reset_cache!(drone_swarm_params, t_data, data, ẍₗ, αₗ, ẍᵢ, step_first)
    # grad1 = FiniteDiff.finite_difference_gradient(p_nn_T_drone-> loss(data_trimmed, t_data_trimmed, drone_swarm_params, u0, p_nn_T_drone), p_nn_T_drone)

    # Method 2 - FiniteDifferences # Why get different results for FiniteDiff and FiniteDifferences???
    # reset_cache!(drone_swarm_params, t_data, data, ẍₗ, αₗ, ẍᵢ, step_first)
    # ab2 = AD.FiniteDifferencesBackend() 
    # grad2 = AD.gradient(ab2, p_nn_T_drone-> loss(data_trimmed, t_data_trimmed, drone_swarm_params, u0, p_nn_T_drone), p_nn_T_drone)


    # Method 3 - ForwardDiff. ERROR: "Cannot determine ordering of Dual tags Nothing and ForwardDiff.Tag{var"#201#204", Float32}". 
    # Might need to flatten and unflatten matricies in ODE function to make work???
    # reset_cache!(drone_swarm_params, t_data, data, ẍₗ, αₗ, ẍᵢ, step_first)
    # grad3 = ForwardDiff.gradient(p_nn_T_drone-> loss(data_trimmed, t_data_trimmed, drone_swarm_params, u0, p_nn_T_drone), p_nn_T_drone)
    # ab3 = AD.ForwardDiffBackend()
    # grad3 = AD.gradient(ab3, p_nn_T_drone-> loss(data_trimmed, t_data_trimmed, drone_swarm_params, u0, p_nn_T_drone), p_nn_T_drone)
    
    # Method 4 - ReverseDiff. ERROR: "type TrackedArray has no field u"
    # reset_cache!(drone_swarm_params, t_data, data, ẍₗ, αₗ, ẍᵢ, step_first)
    # ab4 = AD.ReverseDiffBackend()
    # grad4 = AD.gradient(ab4, p_nn_T_drone-> loss(data_trimmed, t_data_trimmed, drone_swarm_params, u0, p_nn_T_drone), p_nn_T_drone)
    
    # Method 5 - Zygote. ERROR: "Cannot determine ordering of Dual tags Nothing and ForwardDiff.Tag{ODEFunction{true, SciMLBase.AutoSpecialize, DroneSwarmParams, UniformScaling{Bool}, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, typeof(SciMLBase.DEFAULT_OBSERVED), Nothing, Nothing}, Float64}"
    # reset_cache!(drone_swarm_params, t_data, data, ẍₗ, αₗ, ẍᵢ, step_first)
    # ab5 = AD.ZygoteBackend()
    # grad5 = AD.gradient(ab5, p_nn_T_drone-> loss(data_trimmed, t_data_trimmed, drone_swarm_params, u0, p_nn_T_drone), p_nn_T_drone)

    # Method 6 - Tracker ERROR: "type TrackedArray has no field u"
    # reset_cache!(drone_swarm_params, t_data, data, ẍₗ, αₗ, ẍᵢ, step_first)
    # ab6 = AD.TrackerBackend()
    # grad6 = AD.gradient(ab6, p_nn_T_drone-> loss(data_trimmed, t_data_trimmed, drone_swarm_params, u0, p_nn_T_drone), p_nn_T_drone)
    

end


# TEST: Training
# Train with same ODE simply with tension vectors defined using quasi-static assumption like in paper
# Will later do using real data from simulator
# Note only trained with specific Lambda used above. Could change lambda to train with different data 
begin
    ## prepare_initial
    #reset_cache!(drone_swarm_params, t_data, data, ẍₗ, αₗ, ẍᵢ, step_first)
    
    ## Opt parameters
    lr = 0.01 #0.01

    ## Plot results/prediction before training

    ## Setup and run the optimization
    # Set up training data callable struct to store training information in callback
    maxiters = 10000 #500 #1000 #100
    training_data = TrainingData(0, Vector{Float64}(undef, maxiters+1)) # Use Float64[] and push if don't want to pre-allocate array

    adtype = Optimization.AutoFiniteDiff() 
    optf = Optimization.OptimizationFunction((p_nn_T_drone, p)-> loss(data_trimmed, t_data_trimmed, data, ẍₗ, αₗ, ẍᵢ, t_data, drone_swarm_params, u0, p_nn_T_drone, step_first), adtype) #(x, p) -> loss(x), adtype)

    optprob = Optimization.OptimizationProblem(optf, p_nn_T_drone)
    res = Optimization.solve(optprob, OptimizationOptimisers.Adam(lr), maxiters = maxiters, callback = training_data) #TODO: Can we cancel after loss is small enough?? 
    
    # Update nn params based on training
    p_nn_T_drone .= res


    ## Plot loss history
    plot_loss(training_data)

    ## Plot results/prediction after training

end


# PLOT TRAINED RESULTS ALONGSIDE TRAINING DATA
begin
    # Select trimmed or untrimmed data
    t_data_plot = t_data_trimmed #t_data
    data_plot = data_trimmed #data

    # Plot prediction vs data
    # Prior to training
    plot_pred_vs_data(t_data_plot, data_plot, ẍₗ_trimmed, αₗ_trimmed, ẍᵢ_trimmed, αᵢ_trimmed, p_nn_T_drone_pre_train, drone_swarm_params, u0, data, ẍₗ, αₗ, ẍᵢ, αᵢ, t_data, step_first)

    # Post training
    plot_pred_vs_data(t_data_plot, data_plot, ẍₗ_trimmed, αₗ_trimmed, ẍᵢ_trimmed, αᵢ_trimmed, p_nn_T_drone, drone_swarm_params, u0, data, ẍₗ, αₗ, ẍᵢ, αᵢ, t_data, step_first)

end


# TODO:
# Do I need to do data batches, epochs etc??
# How to visualize

# epochs = 400
# opt = ADAM(lr)
# list_plots = []
# losses = []


# plot optimized control - not required
#visualization_callback(res.u, loss(res.u); doplot = true)

# TIPS from Frank:
# Make sure that function learn is easily -> Try just setting first element to T_NN to train just this.
# Rember to add hidden layer back in T_nn_drone


# More training attempts
begin

    # AND HERE
    # REMEMBER TO UPDATE PARAMS 
    # Train the neural network using the ADAM optimizer from Flux.jl
    # ps = Flux.params(params_sense) # OR params_sense.T_drone_nn???
    # opt = ADAM(0.01)
    # data = [xᵢ, ẋᵢ, θᵢ, Ωᵢ, xₗ, ẋₗ, θₗ, Ωₗ] # DOES THIS NEED TO BE FLATTENED???

    # Flux.train!(loss, ps, data, opt)

    # # Test your trained neural network - What here takes in trained params???
    # final_sol = solve(prob, Tsit5(), saveat = t_data)

    # Perhaps this will work too. Unlikelu need to use
    # for epoch in 1:epochs
    #     println("epoch: $epoch / $epochs")
    #     #local u0 = prepare_initial(myparameters.dt, myparameters.numtraj) # Do I need random ICs??
    #     # _dy, back = @time Zygote.pullback(p -> loss(p, u0, myparameters,
    #     #     sensealg=InterpolatingAdjoint()), p_nn)
    #     # gs = @time back(one(_dy))[1]

    #     push!(losses, _dy)
    #     Flux.Optimise.update!(opt, p_nn, gs)
    #     println("")
    # end

    # # plot training loss
    # pl = plot(losses, lw = 1.5, xlabel = "some epochs", ylabel="Loss", legend=false)

    # optimize the parameters for a few epochs with ADAM on time span


end
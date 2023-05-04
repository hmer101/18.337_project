# 18.337 Project - Learning cable tension vectors in multi-drone slung load carrying
# 
# Author: Harvey Merton


begin
   using DifferentialEquations 
   using Flux
   using DiffEqFlux
   using DiffEqCallbacks
   using ForwardDiff

   using Rotations
   using LinearAlgebra
   
   using Plots
   using Plots.PlotMeasures
#    using Printf

   include("utils.jl")
   include("datatypes.jl")
end

begin
    const NUM_DRONES = 3 
end

# Function that returns num_drones drone inputs (forces and moments) at a specified time
# Currently set values close to quasi-static state-regulation inputs. TODO: might need to change to include horizonal components 
# TODO: Perhaps try trajectory-following
function drone_forces_and_moments(params, t::Float64)
    # Preallocate the output vector
    fₘ = [[0.0, Vector{Float64}(undef, 3)] for i in 1:params.num_drones]
    
    # Store the force and moment inputs for each drone
    for i in 1:params.num_drones
        fₘ[i][1] = (params.m_drones[i]+params.m_cables[i])*params.g + params.m_load/params.num_drones # Force
        fₘ[i][2] = [0.0, 0.0, 0.0] # Moment
    end

    return fₘ
end

# u = x = []
# TODO: might need to flatten and unflatten matricies if using ForwardDiff
# TODO: replace all du= with du.=
function (params::DroneSwarmParams)(du,u,p,t) #ode_sys_drone_swarm_nn!(du,u,p,t)
    # Get force and moment inputs from drones
    fₘ = drone_forces_and_moments(params, t)      # TODO: FIX THIS -> how coming through in training data??? Output force required in training data then generate this force for each drone in the generate_forces_and_moments function   Perhaps set a force in training data rather than desired load trajectory?? Or simply use desired load trajectory formulation here w/o force

    ## Variable unpacking 
    e₃ = [0,0,1] 

    # Load
    xₗ = u[params.num_drones*4+1]
    ẋₗ = u[params.num_drones*4+2]
    θₗ = u[params.num_drones*4+3]
    Ωₗ = u[params.num_drones*4+4]

    Rₗ = rpy_to_R([θₗ[1], θₗ[2], θₗ[3]]) # RPY angles to rotation matrix

    ## Equations of motion
    # Accumulate effect of tension on load from all drones 
    ∑RₗTᵢ_load = zeros(3)
    ∑rᵢxTᵢ_load = zeros(3)

    # All drones
    for i in 1:params.num_drones
        ### Variable unpacking
        # Drone states
        xᵢ = u[i]
        ẋᵢ = u[params.num_drones+i]
        θᵢ = u[2*params.num_drones+i]
        Ωᵢ = u[3*params.num_drones+i] # Same as θ̇ᵢ

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
        # Estimate accelerations with backwards FD to allow cable tensions to be calculated
        t_step = t - params.t_prev # TODO: IF t_step=0, assign all estimates =0 to account for starting step (otherwise will get /0 error)
        ẍₗ_est = (ẋₗ-params.ẋₗ_prev)/t_step
        αₗ_est = (Ωₗ-params.Ωₗ_prev)/t_step
        ẍᵢ_est = (ẋᵢ-params.ẋᵢ_prev[i])/t_step

        ẍ₍Lᵢ₎ = ẍₗ_est + cross(αₗ_est, params.r_cables[i]) + cross(Ωₗ, cross(Ωₗ, params.r_cables[i]))
        ẍ₍i_rel_Lᵢ₎ = ẍᵢ_est - ẍ₍Lᵢ₎ - cross(αₗ_est, r₍i_rel_Lᵢ₎) - cross(Ωₗ, cross(Ωₗ,r₍i_rel_Lᵢ₎)) - 2*cross(Ωₗ,ẋ₍i_rel_Lᵢ₎)
 

        # # Drone side
        nn_ip = vcat(x₍i_rel_Lᵢ₎, ẋ₍i_rel_Lᵢ₎, ẍ₍i_rel_Lᵢ₎)
        nn_ip = convert.(Float32, nn_ip)

        Tᵢ_drone =    #p.T_drone_nn(nn_ip)
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
        du[i] = ẋᵢ

        # Acceleration
        ẍᵢ = (1/params.m_drones[i])*(fᵢ*Rᵢ*e₃ - params.m_drones[i]*params.g*e₃ + Rₗ*Tᵢ_drone) # Might not use Rl when not using assumption?????
        du[i+params.num_drones] = ẍᵢ

        # Angular velocity
        du[i+2*params.num_drones] = Ωᵢ

        # Angular acceleration
        # αᵢ = inv(p.j_drones[i])*(mᵢ - cross(Ωᵢ,(p.j_drones[i]*Ωᵢ)))
        du[i+3*params.num_drones] = inv(params.j_drones[i])*(mᵢ - cross(Ωᵢ,(params.j_drones[i]*Ωᵢ)))
        

        ### Update cache
        params.ẋᵢ_prev[i] = ẋᵢ

    end

    ## Load EOM
    # Velocity
    du[1+4*params.num_drones] = ẋₗ

    # Acceleration
    ẍₗ = (1/params.m_load)*(-∑RₗTᵢ_load-params.m_load*params.g*e₃)
    du[2+4*params.num_drones] = ẍₗ

    # Angular velocity
    du[3+4*params.num_drones] = Ωₗ

    # Angular acceleration
    αₗ = inv(params.j_load)*(∑rᵢxTᵢ_load - cross(Ωₗ,(params.j_load*Ωₗ)))
    du[4+4*params.num_drones] = αₗ

    #du = ArrayPartition[] # RecursiveArrayTools

    ### Update cache
    params.ẋₗ_prev = ẋₗ
    params.Ωₗ_prev = Ωₗ
    params.t_prev = t

end

# function print_step_size(integrator)
#     step_size = integrator.t - integrator.tprev
#     println("Current time: $(integrator.t), Step size: $step_size")
# end

# Find l2 loss 
# function loss(data, t_data, sol)
#     # if size(y_true) != size(y_pred)
#     #     throw(ArgumentError("y_true and y_pred must have the same dimensions"))
#     # end

#     l = 0
#     for i in 1:size(data,2)
#         l += sum((data[i] - sol(t_data[i])).^2)
#     end
#     return l
# end

function loss(p, data, t_data) # data, t_data, sol)
    sol = 
    
    
    l = 0
    for i in 1:size(data,2)
        l += sum((data[i] - sol(t_data[i])).^2) # PERHAPS NEED TO FLATTEN???? ArrayToolsArray
    end
    return l
end


# Function that returns num_drones tension vectors for the massless cable case
# TODO: Can this work for non-massless case?
function cable_tensions_massless_cable(ẍₗ, Ωₗ, αₗ, Rₗ, params)
    e₃ = [0,0,1] 

    # Create matricies required for Euler-Newton tension calculations
    # ϕ is I's and hat mapped r's for transforming wrench to tensions
    # N is the kernal of ϕ
    ϕ = [Matrix{Float64}(I, params.num_drones, params.num_drones) for _ in 1:2, _ in 1:params.num_drones] # 2xn matrix of I matricies #Matrix{Float64}(undef, 2*params.num_drones, 3*params.num_drones)
    #num_N_rows = 3*params.num_drones
    N = [zeros(params.num_drones) for _ in 1:params.num_drones, _ in 1:params.num_drones] #nxn matrix of vectors #Matrix{Float64}(undef, num_N_rows, params.num_drones)
    n_col = 1

    # Loop through all cable attachment points
    for i in 1:params.num_drones
        # Generate ϕ column by column
        ϕ[2,i] = hat_map(params.r_cables[i])

        # Generate N column by column
        for j in i:params.num_drones #i is cols, j is rows
            if i != j
                # Unit vector from ith cable's attachement point to jth cable's attachment point
                r₍j_rel_i₎ = params.r_cables[j] - params.r_cables[i]
                u₍ij₎ = r₍j_rel_i₎/norm(r₍j_rel_i₎)
                
                # Generate next N column using unit vectors between cable attachment points on load
                N_next_col = [zeros(3) for _ in 1:params.num_drones]

                N_next_col[i] = u₍ij₎
                N_next_col[j] = -u₍ij₎
                N[:,n_col] = N_next_col

                n_col +=1
            end
        end
    end

    # Wrench calculations
    W = -(vcat(transpose(Rₗ)*(params.m_load*(ẍₗ + params.g*e₃)), (params.j_load*αₗ + cross(Ωₗ, params.j_load*Ωₗ))))

    # Internal force components (defined here to produce zero isometric work)
    Λ = zeros(params.num_drones)
    
    # Calculate T's - returns as a vector of tension 3-vectors
    T = unflatten_v(pinv(flatten_m(ϕ))*W + flatten_m(N)*Λ, 3)
    #T = -T # To change vector direction so pointing from load to drone

    return T
end

# Generate training data using differential flatness and 
function generate_tension_training_data(t_step, params)
    # Use a circlular trajectory for the load with constant scalar velocity at walking pace 1.2 m/s
    # Will change with selected load trajectory
    v_target = 5.0 #1.2
    r_scalar = 2.0 # Radius of circular trajectory

    ω_scalar = v_target/r_scalar
    ω = [0.0, 0.0, ω_scalar]

    # Set time to capture 1 full rotation
    T = 2*π/ω_scalar 
    t_data = 0.0:t_step:T

    ## Generate the circular load trajectory
    # Preallocate arrays
    # Drones
    xᵢ = [[Vector{Float64}(undef, 3) for _ in 1:params.num_drones] for _ in 1:length(t_data)] 
    ẋᵢ = [[Vector{Float64}(undef, 3) for _ in 1:params.num_drones] for _ in 1:length(t_data)] 
    ẍᵢ = [[Vector{Float64}(undef, 3) for _ in 1:params.num_drones] for _ in 1:length(t_data)]

    θᵢ = [[Vector{Float64}(undef, 3) for _ in 1:params.num_drones] for _ in 1:length(t_data)] 
    Ωᵢ = [[Vector{Float64}(undef, 3) for _ in 1:params.num_drones] for _ in 1:length(t_data)] 
    αᵢ = [[Vector{Float64}(undef, 3) for _ in 1:params.num_drones] for _ in 1:length(t_data)]

    # Load
    xₗ = [Vector{Float64}(undef, 3) for _ in 1:length(t_data)]
    ẋₗ = [Vector{Float64}(undef, 3) for _ in 1:length(t_data)]
    ẍₗ = [Vector{Float64}(undef, 3) for _ in 1:length(t_data)]

    θₗ = [Vector{Float64}(undef, 3) for _ in 1:length(t_data)]
    Ωₗ = [Vector{Float64}(undef, 3) for _ in 1:length(t_data)]
    αₗ = [Vector{Float64}(undef, 3) for _ in 1:length(t_data)]

    # Drone relative to load
    # x₍i_rel_Lᵢ₎ = [[Vector{Float64}(undef, 3) for _ in 1:params.num_drones] for _ in 1:length(t_data)] 
    # ẋ₍i_rel_Lᵢ₎ = [[Vector{Float64}(undef, 3) for _ in 1:params.num_drones] for _ in 1:length(t_data)] 
    # ẍ₍i_rel_Lᵢ₎ = [[Vector{Float64}(undef, 3) for _ in 1:params.num_drones] for _ in 1:length(t_data)] 
    T = [[Vector{Float64}(undef, 3) for _ in 1:params.num_drones] for _ in 1:length(t_data)] 

    # Cache previous values for FD calculations
    xᵢ_prev = [[0.0, 0.0, 0.0] for _ in 1:params.num_drones] # Could initialise at starting values so to not have wild derivatives at start
    ẋᵢ_prev = [[0.0, 0.0, 0.0] for _ in 1:params.num_drones] # Will have wild 1st derivative for 1st step and wild second for first 2 steps with 0 ICs
    θᵢ_prev = [[0.0, 0.0, 0.0] for _ in 1:params.num_drones]
    Ωᵢ_prev = [[0.0, 0.0, 0.0] for _ in 1:params.num_drones]

    # Use a rotating co-ordinate system that rotates with ω
    θₜ = 0

    # Loops are just as fast as vectorisation in Julia
    for (ind, t) in enumerate(t_data)
        
        # Set position and derivatives - will change with selected load trajectory
        xₗ[ind] = [r_scalar*cos(θₜ), r_scalar*sin(θₜ), 0.0]
        ẋₗ[ind] = cross(ω, xₗ[ind])
        ẍₗ[ind] = cross(ω, cross(ω, xₗ[ind]))

        # Set angular velocity and derivatives - will change with selected load trajectory
        θₗ[ind] = [0.0, 0.0, 0.0]
        Ωₗ[ind] = [0.0, 0.0, 0.0]
        αₗ[ind] = [0.0, 0.0, 0.0]

        Rₗ = rpy_to_R(θₗ[ind])
        T₍ind₎ = cable_tensions_massless_cable(ẍₗ[ind], Ωₗ[ind], αₗ[ind], Rₗ, params)
        T[ind] = T₍ind₎

        ## Calculate drone positions (and derivatives) relative to load attachment point
        for i in 1:params.num_drones
            ## Drone relative to world
            # Position
            qᵢ = T₍ind₎[i]/norm(T₍ind₎[i])
            Lᵢqᵢ = params.l_cables[i]*qᵢ 

            xᵢ[ind][i] = xₗ[ind] + Rₗ*(params.r_cables[i]-Lᵢqᵢ)
            
            # Velocity (use backwards finite difference)
            ẋᵢ[ind][i] = (xᵢ[ind][i]-xᵢ_prev[i])/t_step

            # Acceleration (use second order backwards finite difference)
            ẍᵢ[ind][i] = (ẋᵢ[ind][i]-ẋᵢ_prev[i])/t_step
            
            # Orientation (will change with different trajectories)
            θᵢ[ind][i] = [0.0, 0.0, 0.0]

            # Angular velocity 
            Ωᵢ[ind][i] = (θᵢ[ind][i] - θᵢ_prev[i])/t_step

            # Angular acceleration
            αᵢ[ind][i] = (Ωᵢ[ind][i] - Ωᵢ_prev[i])/t_step

            ## Drone relative to attachment point
            # Position
            # r₍i_rel_Lᵢ₎ = -Lᵢqᵢ
            # x₍i_rel_Lᵢ₎[ind][i] = r₍i_rel_Lᵢ₎
            
            # # Velocity
            # ẋ₍i_rel_Lᵢ₎[ind][i] = ẋᵢ[ind][i] - (ẋₗ[ind] + cross(Ωₗ[ind], params.r_cables[i]))

            # # Acceleration
            # ẍ₍Lᵢ₎ = ẍₗ[ind] + cross(αₗ[ind],params.r_cables[i]) + cross(Ωₗ[ind], cross(Ωₗ[ind], params.r_cables[i]))
            # ẍ₍i_rel_Lᵢ₎[ind][i] = ẍᵢ[ind][i] - ẍ₍Lᵢ₎ - cross(αₗ[ind],r₍i_rel_Lᵢ₎) - cross(Ωₗ[ind], cross(Ωₗ[ind],r₍i_rel_Lᵢ₎)) - 2*cross(Ωₗ[ind],ẋ₍i_rel_Lᵢ₎[ind][i])

            ## Update values for backwards finite difference
            xᵢ_prev[i] = xᵢ[ind][i]
            ẋᵢ_prev[i] = ẋᵢ[ind][i]
            θᵢ_prev[i] = θᵢ[ind][i]
            Ωᵢ_prev[i] = Ωᵢ[ind][i]

        end

        # Update load position
        θₜ = θₜ + t_step*ω_scalar

    end

    # Return time vector, tension data, load trajectory and drone motion                   # OLD: relative to cable attachment points on load
    return t_data, xᵢ, ẋᵢ, ẍᵢ, θᵢ, Ωᵢ, αᵢ, xₗ, ẋₗ, ẍₗ, θₗ, Ωₗ, αₗ  #T, x₍i_rel_Lᵢ₎, ẋ₍i_rel_Lᵢ₎, ẍ₍i_rel_Lᵢ₎

end



begin
    ## Set initial conditions
    u0 = [Vector{Float64}(undef, 3) for i in 1:(6*NUM_DRONES+4)]
    
    # Add drones ICs
    for i in 1:NUM_DRONES
        ## Drones
        # Position
        u0[i] = i*ones(3)

        # Velocity
        u0[NUM_DRONES+i] = i*ones(3)

        # Orientation
        u0[2*NUM_DRONES+i] = i*ones(3)

        # Angular velocity
        u0[3*NUM_DRONES+i] = i*ones(3)

    end
    # Load ICs
    # Position
    u0[1+4*NUM_DRONES] = 100*ones(3)

    # Velocity
    u0[2+4*NUM_DRONES] = 100*ones(3)

    # Orientation
    u0[3+4*NUM_DRONES] = 100*ones(3)

    # Angular velocity
    u0[4+4*NUM_DRONES] = 100*ones(3)


    ## Setup parameters
    # Cable tension NNs - take in flattened vector inputs, output tension vector at drone and load respectively
    input_dim = 9 # Position, velocity and acceleration vectors for each drone relative to the attachment point of their attached cables on the load
    nn_T_dot_drone = Chain(Dense(input_dim, 32, tanh), Dense(32, 3)) # TODO: Currently 1 hidden layer - could try 2!!
    nn_T_dot_load = Chain(Dense(input_dim, 32, tanh), Dense(32, 3))

    # Initialise parameter struct
    # Note that cache values are initialised at corresponding u0 values
    j_drone = [2.32 0 0; 0 2.32 0; 0 0 4]

    # drone_swarm_params = DroneSwarmParams_init(num_drones=NUM_DRONES, g=9.81, m_load=0.225, m_drones=[0.5, 0.5, 0.5], m_cables=[0.1, 0.1, 0.1], l_cables=[1.0, 1.0, 1.0],
    #                                 j_load = [2.1 0 0; 0 1.87 0; 0 0 3.97], j_drones= [j_drone, j_drone, j_drone], 
    #                                 r_cables = [[-0.42, -0.27, 0], [0.48, -0.27, 0], [-0.06, 0.55, 0]], t_prev=0.1, ẋₗ_prev=u0[2+4*NUM_DRONES], Ωₗ_prev=u0[4+4*NUM_DRONES], ẋᵢ_prev=u0[NUM_DRONES+1:2*NUM_DRONES]) #, T_dot_drone_nn=nn_T_dot_drone, T_dot_load_nn=nn_T_dot_load)

    drone_swarm_params = DroneSwarmParams(9.81, NUM_DRONES, 0.225, [0.5, 0.5, 0.5], [0.1, 0.1, 0.1], [1.0, 1.0, 1.0], [2.1 0 0; 0 1.87 0; 0 0 3.97], [j_drone, j_drone, j_drone], [[-0.42, -0.27, 0], [0.48, -0.27, 0], [-0.06, 0.55, 0]], 0.0, u0[2+4*NUM_DRONES], u0[4+4*NUM_DRONES], u0[NUM_DRONES+1:2*NUM_DRONES])

end

begin
   
    ## Generate training data
    t_step_data = 0.1
    #t_data, xᵢ, ẋᵢ, ẍᵢ, θᵢ, Ωᵢ, αᵢ, xₗ, ẋₗ, ẍₗ, θₗ, Ωₗ, αₗ = generate_tension_training_data(t_step_data, drone_swarm_params)

    # Display trajectory - load
    # plot_trajectory(t_data, xₗ, ẋₗ, ẍₗ, true, false)
    # plot_trajectory(t_data, θₗ, Ωₗ, αₗ, true, true)

    # Display trajectory - drone
    # plot_trajectory(t_data[3:end], xᵢ[3:end], ẋᵢ[3:end], ẍᵢ[3:end], false, false)
    # plot_trajectory(t_data, θᵢ[3:end], Ωᵢ[3:end], αᵢ[3:end], false, true)


    T_drone_nn = Chain(Dense(9, 32, tanh), Dense(32, 3))
    params_sense = DroneSwarmParamsSense(T_drone_nn)

    du = [Vector{Float64}(undef, 3) for i in 1:(4*drone_swarm_params.num_drones+4)]
    t_span = (0.0, 1.0)
    time_save_points = t_span[1]:t_step_data:t_span[2]
    #time_points = 

    # Test one function call
    # println("before: $du")
    # drone_swarm_params(du, u0, params_sense, t) # Some NaNs
    # println("after: $du")


    # Run ODE solver (define better u0's, proper fi's and mi's !)
    #step_size_callback = DiscreteCallback(true, print_step_size) # TODO: Tune this (when save etc.)

    # WANT HELP HERE - put all into loss function!!! Need to recreate ODEProblem every time
    p = Flux.destructure() # TODO: make this work (AS WELL AS ARRAY TOOLS)
    prob = ODEProblem(drone_swarm_params, u0, t_span, p) # HOW DO THE NN PARAMS GET IN HERE??? 
    sol = solve(prob, Tsit5(), saveat=time_save_points, abstol = 1e-12, reltol = 1e-12) # MAKE SURE TO USE CORRECT STEP SIZE IN ODE callback=step_size_callback, OR used fixed timestep solve dt=myparameters.dt??


    
    # FiniteDiff.finite_difference_gradient() # Easier
    # ForwardDiff.gradient(p-> loss(p, data, t_data), p) # Use flattened params p here


    ## Train
    # Train with same ODE simply with tension vectors defined using quasi-static assumption like in paper
    # Will later do using real data from simulator
    # Note only trained with specific Lambda used above. Could change lambda to train with different data 

    # AND HERE
    # Train the neural network using the ADAM optimizer from Flux.jl
    # ps = Flux.params(params_sense) # OR params_sense.T_drone_nn???
    # opt = ADAM(0.01)
    # data = [xᵢ, ẋᵢ, θᵢ, Ωᵢ, xₗ, ẋₗ, θₗ, Ωₗ] # DOES THIS NEED TO BE FLATTENED???

    # Flux.train!(loss, ps, data, opt)

    # # Test your trained neural network - What here takes in trained params???
    # final_sol = solve(prob, Tsit5(), saveat = t_data)






    # optimize the parameters for a few epochs with ADAM on time span Nint
    # lr = 0.01
    # epochs = 400

    # opt = ADAM(lr)
    # list_plots = []
    # losses = []

    # FiniteDiff.finite_difference_gradient() # Easier
    # ForwardDiff.gradient(p-> loss(p, data, t_data), p) # Use flattened params p here


     # # Setup and run the optimization - USE THIS NEXT!!!!
    # adtype = Optimization.AutoZygote() # Could try "AutoForward" or someting else - Finite??
    # optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)


    # optprob = Optimization.OptimizationProblem(optf, p_nn)
    # res = Optimization.solve(optprob, OptimizationOptimisers.Adam(myparameters.lr), maxiters = 100) #callback = visualization_callback,                      


    # # plot optimized control - not required
    # visualization_callback(res.u, loss(res.u); doplot = true)







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
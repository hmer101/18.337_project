# 18.337 Project - Learning cable tension vectors in multi-drone slung load carrying
# 
# Author: Harvey Merton


begin
   using DifferentialEquations 
   using Flux
   using DiffEqFlux

   using Rotations
   using LinearAlgebra
   
   using Plots
   using Plots.PlotMeasures
   using Printf

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
function ode_sys_drone_swarm_nn!(du,u,p,t)
    # Get force and moment inputs from drones
    fₘ = drone_forces_and_moments(p, t)      # TODO: FIX THIS -> how coming through in training data???

    ## Variable unpacking 
    e₃ = [0,0,1] 

    # Load
    xₗ = u[p.num_drones*4+1]
    ẋₗ = u[p.num_drones*4+2]
    θₗ = u[p.num_drones*4+3]
    Ωₗ = u[p.num_drones*4+4]

    Rₗ = rpy_to_R([θₗ[1], θₗ[2], θₗ[3]]) # RPY angles to rotation matrix



    ## Drone relative to attachment point
    # Position
    # r₍i_rel_Lᵢ₎ = -Lᵢqᵢ
    # x₍i_rel_Lᵢ₎[ind][i] = r₍i_rel_Lᵢ₎
    
    # # Velocity
    # ẋ₍i_rel_Lᵢ₎[ind][i] = ẋᵢ[ind][i] - (ẋₗ[ind] + cross(Ωₗ[ind], params.r_cables[i]))

    # # Acceleration
    # ẍ₍Lᵢ₎ = ẍₗ[ind] + cross(αₗ[ind],params.r_cables[i]) + cross(Ωₗ[ind], cross(Ωₗ[ind], params.r_cables[i]))
    # ẍ₍i_rel_Lᵢ₎[ind][i] = ẍᵢ[ind][i] - ẍ₍Lᵢ₎ - cross(αₗ[ind],r₍i_rel_Lᵢ₎) - cross(Ωₗ[ind], cross(Ωₗ[ind],r₍i_rel_Lᵢ₎)) - 2*cross(Ωₗ[ind],ẋ₍i_rel_Lᵢ₎[ind][i])


    ## Equations of motion
    ## Load
    ∑RₗTᵢ_load = zeros(3)
    ∑rᵢxTᵢ_load = zeros(3)

    # Calculate cumulative effect of cables on load
    for i in 1:p.num_drones
        Tᵢ_drone = u[4*p.num_drones + 4 + i]   # TODO: Note this Ti_load = -Ti_drone relationship will not hold without assumption
        Tᵢ_load = u[5*p.num_drones + 4 + i]

        # Use massless assumption
        if p.use_nn == false
            # Check massless cables, quasi-static assumption - tension vectors on drone and load are equal and opposite
            if !are_opposite_directions(Tᵢ_drone, Tᵢ_load)
                error("Tension vectors do not meet massless cable assumption")
            end

        end

        # Sum across all cables needed for load EOM calculations
        ∑RₗTᵢ_load += Rₗ*Tᵢ_load # Forces
        ∑rᵢxTᵢ_load += cross(p.r_cables[i],-Tᵢ_load) # Moments

    end
    # HEREEE
        #x_Dᵢ_rel_Lᵢ = inv(Rₗ)*(xᵢ - xₗ) - p.r_cables[i] # = -qᵢ
        qᵢ = -inv(Rₗ)*(xᵢ - xₗ) - p.r_cables[i]



    # Load EOM
    # Velocity
    du[1+4*p.num_drones] = ẋₗ

    # Acceleration
    ẍₗ = (1/p.m_load)*(-∑RₗTᵢ_load-p.m_load*p.g*e₃)
    du[2+4*p.num_drones] = ẍₗ

    # Angular velocity
    du[3+4*p.num_drones] = Ωₗ #R_L_dot

    # Angular acceleration
    αₗ = inv(p.j_load)*(∑rᵢxTᵢ_load - cross(Ωₗ,(p.j_load*Ωₗ)))
    du[4+4*p.num_drones] = αₗ


    # All drones
    for i in 1:p.num_drones
        ### Variable unpacking
        # Drone states
        xᵢ = u[i]
        ẋᵢ = u[p.num_drones+i]
        θᵢ = u[2*p.num_drones+i]
        Ωᵢ = u[3*p.num_drones+i] # Same as θ̇ᵢ

        Rᵢ = rpy_to_R([θᵢ[1], θᵢ[2], θᵢ[3]]) # RPY angles to rotation matrix. 

        # Connections (after drone and load in u)
        Tᵢ_drone = u[4*p.num_drones + 4 + i]
        #Tᵢ_load = u[5*p.num_drones + 4 + i]

        # Inputs 
        fᵢ = fₘ[i][1]
        mᵢ = fₘ[i][2]


        ### Equations of motion
        ## Drones
        # Velocity
        du[i] = ẋᵢ

        # Acceleration
        ẍᵢ = (1/p.m_drones[i])*(fᵢ*Rᵢ*e₃ - p.m_drones[i]*p.g*e₃ + Tᵢ_drone) #ORIENTATION NOT DEFINED BY R_L??? R_L*T_i_drone). Should the e₃ after fᵢ*Rᵢ be there???
        du[i+p.num_drones] = ẍᵢ

        # Angular velocity
        du[i+2*p.num_drones] = Ωᵢ

        # Angular acceleration
        # αᵢ = inv(p.j_drones[i])*(mᵢ - cross(Ωᵢ,(p.j_drones[i]*Ωᵢ)))
        du[i+3*p.num_drones] = inv(p.j_drones[i])*(mᵢ - cross(Ωᵢ,(p.j_drones[i]*Ωᵢ)))

        ## Connection (note these come after load indicies to make it easier to change if required)    
        # Drone motion relative to associated cable's connection point on load (for tension vector neural network)



        # HEREREEEEE - Make so can swap out for massless case for generating training data



        x_Dᵢ_rel_Lᵢ = inv(Rₗ)*(xᵢ - xₗ) - p.r_cables[i]
        ẋ_Dᵢ_rel_Lᵢ = ẋᵢ - (ẋₗ + cross(Ωₗ,p.r_cables[i]))

        ẍ_Lᵢ = ẍₗ + cross(αₗ, p.r_cables[i]) + cross(Ωₗ, cross(Ωₗ, p.r_cables[i])) # Acceleration of point on load where cable is attached
        ẍ_Dᵢ_rel_Lᵢ = ẍᵢ - ẍ_Lᵢ - cross(αₗ, x_Dᵢ_rel_Lᵢ) - cross(Ωₗ, cross(Ωₗ,x_Dᵢ_rel_Lᵢ)) - 2*cross(Ωₗ,ẋ_Dᵢ_rel_Lᵢ)
        
        # Drone side
        nn_ip = vcat(x_Dᵢ_rel_Lᵢ, ẋ_Dᵢ_rel_Lᵢ, ẍ_Dᵢ_rel_Lᵢ)
        nn_ip = convert.(Float32, nn_ip)

        du[i+4*p.num_drones+4] = p.T_dot_drone_nn(nn_ip)

        # Load side
        du[i+5*p.num_drones+4] = p.T_dot_load_nn(nn_ip)

    end

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
    
    # Add drones and cables ICs
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

        ## Cables
        # Drone side
        #u0[4*NUM_DRONES + 4 + i] = i*ones(3)

        # Load side
        #u0[5*NUM_DRONES + 4 + i] = i*ones(3)

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
    j_drone = [2.32 0 0; 0 2.32 0; 0 0 4]

    params = DroneSwarmParams_init(num_drones=NUM_DRONES, g=9.81, m_load=0.225, m_drones=[0.5, 0.5, 0.5], m_cables=[0.1, 0.1, 0.1], l_cables=[1.0, 1.0, 1.0],
                                    j_load = [2.1 0 0; 0 1.87 0; 0 0 3.97], j_drones= [j_drone, j_drone, j_drone], 
                                    r_cables = [[-0.42, -0.27, 0], [0.48, -0.27, 0], [-0.06, 0.55, 0]], T_dot_drone_nn=nn_T_dot_drone, T_dot_load_nn=nn_T_dot_load)

end

begin
    # TEST!!!!!!!!!
    ## Solve
    # du = [Vector{Float64}(undef, 3) for i in 1:(6*NUM_DRONES+4)]
    # t = 1.0
    # # print(typeof(du))
    # # print(typeof(u0))
    # #println(u0)

    # ode_sys_drone_swarm_nn!(du,u0,params,t)

    ## Generate training data
    t_data, xᵢ, ẋᵢ, ẍᵢ, θᵢ, Ωᵢ, αᵢ, xₗ, ẋₗ, ẍₗ, θₗ, Ωₗ, αₗ = generate_tension_training_data(0.1, params)


    # Display trajectory - load
    # plot_trajectory(t_data, xₗ, ẋₗ, ẍₗ, true, false)
    # plot_trajectory(t_data, θₗ, Ωₗ, αₗ, true, true)

    # Display trajectory - drone
    plot_trajectory(t_data[3:end], xᵢ[3:end], ẋᵢ[3:end], ẍᵢ[3:end], false, false)
    plot_trajectory(t_data, θᵢ[3:end], Ωᵢ[3:end], αᵢ[3:end], false, true)



    # Display training tension and drone relative to load connection points 
    #plot_results(t_data, T, x₍i_rel_Lᵢ₎, ẋ₍i_rel_Lᵢ₎, ẍ₍i_rel_Lᵢ₎, true)

    ## Define the neural ODE and solve
    #t_data[1], t_data[end]

    # nn_T_dot_drone_test = Chain(Dense(3, 32, tanh), Dense(32, 3)) #Chain(Dense(9, 32, tanh), Dense(32, 3))
    # T_drone_n_ode = NeuralODE(nn_T_dot_drone_test, (Float32(t_data[1]), Float32(t_data[end])), Tsit5(), saveat = 0.1, reltol=1e-7, abstol=1e-9) #(0.0f0, 1.0f0)

    # u0 = Float32[T[1][1][1], T[1][1][2], T[1][1][3]] #0.0, 0.0, 0.0] #0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # pred = T_drone_n_ode(u0) # Get the prediction using the correct initial condition

    # scatter(t_data,T[1,:],label="data")
    # scatter(t_data,pred[1,:],label="prediction")


    ## Train
    # Train with same ODE simply with tension vectors defined using quasi-static assumption like in paper
    # Will later do using real data from simulator

end
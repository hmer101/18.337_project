# 18.337 Project - Learning cable tension vectors in multi-drone slung load carrying
# 
# Author: Harvey Merton


begin
   using DifferentialEquations 
   using Flux
   using DiffEqFlux
   using DiffEqCallbacks
   using ForwardDiff
   using OptimizationOptimisers
   using FiniteDiff

   using Rotations
   using LinearAlgebra
   using Statistics

   using RecursiveArrayTools
   
   using Plots
   using Plots.PlotMeasures

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
    temp_t = params.t_prev # TODO: REMOVE
    println()
    println("t: $t")
    println("params.t_prev: $temp_t")

    t_step = t - params.t_prev #0.1 #t - params.t_prev 
    # println("t_step: $t_step")

    # if t_step == 0.0
    #     ẍₗ_est = params.ẍₗ_est_prev
    #     αₗ_est = params.αₗ_est_prev

    #     #error("Step size from cached previous step is 0.0")
    #     @warn "Step size from cached previous step is 0.0 - using cached estimates for accelerations"
    #     println("ẍₗ_est: $ẍₗ_est")
    #     println("αₗ_est: $αₗ_est")

    # else
    #     ẍₗ_est = (ẋₗ-params.ẋₗ_prev)/t_step
    #     αₗ_est = (Ωₗ-params.Ωₗ_prev)/t_step
    # end

    # Rather than using backwards finite difference, simply estimate current acceleration as previous actual acceleration
    # Backwards finite difference has lag anyway, this just avoids numerical error amplification by small timesteps
    ẍₗ_est = params.ẍₗ_prev
    αₗ_est = params.αₗ_prev

    println("ẍₗ_est: $ẍₗ_est")
    println("αₗ_est: $αₗ_est")

    # ẍₗ_est = [0.0, 0.0, 0.0] 
    # αₗ_est = [0.0, 0.0, 0.0]
    # ẍₗ_est = (ẋₗ-params.ẋₗ_prev)/t_step
    # αₗ_est = (Ωₗ-params.Ωₗ_prev)/t_step

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
        # t_step = t - params.t_prev 

        # if t_step == 0.0
        #     error("Step size from cached previous step is 0.0")
        # end

        # println()
        # println("t_step")
        # println(t_step)
 
        # If t_step=0, assign all estimates =0 to account for starting step (otherwise will get /0 error)
        # TODO: Could simply save these values calc'ed at previous timestep (in params) rather than use FD -> use delayed estimation
        # TODO: Could go back to zero!! Currently trying setting prev velocities based on actual prev velocities before data start
        # ẍₗ_est = [0.0, 0.0, 0.0] 
        # αₗ_est = [0.0, 0.0, 0.0]
        #ẍᵢ_est = [0.0, 0.0, 0.0]
        
        # Otherwise, estimate accelerations with backwards FD to allow cable tensions to be calculate

        # if t_step == 0.0
        #     # @warn "Also for drone" # TODO: REMOVE!!
        #     ẍᵢ_est = params.ẍᵢ_est_prev[i]
        #     # println("αₗ_est: $αₗ_est")
        # else
        #     ẍᵢ_est = (ẋᵢ-params.ẋᵢ_prev[i])/t_step
        # end
        ẍᵢ_est = params.ẍᵢ_prev[i]
        println("ẍᵢ_est: $ẍᵢ_est")


        ẍ₍Lᵢ₎ = ẍₗ_est + cross(αₗ_est, params.r_cables[i]) + cross(Ωₗ, cross(Ωₗ, params.r_cables[i]))
        ẍ₍i_rel_Lᵢ₎ = ẍᵢ_est - ẍ₍Lᵢ₎ - cross(αₗ_est, r₍i_rel_Lᵢ₎) - cross(Ωₗ, cross(Ωₗ,r₍i_rel_Lᵢ₎)) - 2*cross(Ωₗ,ẋ₍i_rel_Lᵢ₎)
 

        ## Drone side
        nn_ip = vcat(x₍i_rel_Lᵢ₎, ẋ₍i_rel_Lᵢ₎, ẍ₍i_rel_Lᵢ₎)
        nn_ip = convert.(Float32, nn_ip)

        #println(nn_ip)

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
        params.ẋᵢ_prev[i][1:end] = ẋᵢ[1:end]
        params.ẍᵢ_prev[i][1:end] = ẍᵢ[1:end] #_est

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

    params.ẋₗ_prev[1:end] = ẋₗ[1:end]
    params.Ωₗ_prev[1:end] = Ωₗ[1:end]
    
    params.ẍₗ_prev[1:end] = ẍₗ[1:end] #_est
    params.αₗ_prev[1:end] = αₗ[1:end] #_est

end


function loss(data, t_data, drone_swarm_params, u0, p_nn_T_drone)
    # Solve the ODE
    t_span = (t_data[1], t_data[end])  #t_span = (0.0, 1.0)
    time_save_points = t_span[1]:(t_data[2]-t_data[1]):t_span[2] # Assuming data saved at fixed-distance points round(, digits=3)

    # println("t_span: $t_span") # Perhaps rounding errors here
    # println("time_save_points: $time_save_points")

    #print("drone_swarm_params: $drone_swarm_params")

    prob = ODEProblem(drone_swarm_params, u0, t_span, p_nn_T_drone) # Check that step size in ODE is correct. callback=step_size_callback, OR used fixed timestep solve dt=myparameters.dt??
    sol = solve(prob, Tsit5(), saveat=time_save_points, dt=0.01)
    
    # Get loss relative to data
    l = 0
    for step_num in 1:length(t_data) #size(data,2)
        l += mean((data[step_num] - sol.u[step_num]).^2) #sum rather than mean?? # PERHAPS NEED TO FLATTEN???? ArrayToolsArray
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
    #return t_data, xᵢ, ẋᵢ, ẍᵢ, θᵢ, Ωᵢ, αᵢ, xₗ, ẋₗ, ẍₗ, θₗ, Ωₗ, αₗ  #T, x₍i_rel_Lᵢ₎, ẋ₍i_rel_Lᵢ₎, ẍ₍i_rel_Lᵢ₎

    ## Construct an array of array partitions containing training data (TODO: Modify for changing number of drones. Change so start with ArrayPartition rather than convert after)
    # Preallocate data array. TODO: Must be better way
    ap_temp = ArrayPartition([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    data = [copy(ap_temp) for _ in 1:length(t_data)]

    # Assign elements of data array
    # Across all timesteps
    for i in 1:length(t_data)
        # Across all components
        data[i] = ArrayPartition(xᵢ[i][1], xᵢ[i][2], xᵢ[i][3], ẋᵢ[i][1], ẋᵢ[i][2], ẋᵢ[i][3], 
                                θᵢ[i][1], θᵢ[i][2], θᵢ[i][3], Ωᵢ[i][1], Ωᵢ[i][2], Ωᵢ[i][3], 
                                xₗ[i], ẋₗ[i], θₗ[i], Ωₗ[i])
        #data[i].x[1][1:length(arr_parts[1].x[1])] = [2.0, 2.0, 2.0]
    end

    return t_data, data, ẍᵢ, αᵢ, ẍₗ, αₗ

end


# SET UP IC's and PARAMS
begin
    ## Set initial conditions
    u0_temp = [Vector{Float64}(undef, 3) for i in 1:(6*NUM_DRONES+4)]
    
    # Add drones ICs
    for i in 1:NUM_DRONES
        ## Drones
        # Position
        u0_temp[i] = i*ones(3)

        # Velocity
        u0_temp[NUM_DRONES+i] = i*ones(3)

        # Orientation
        u0_temp[2*NUM_DRONES+i] = i*ones(3)

        # Angular velocity
        u0_temp[3*NUM_DRONES+i] = i*ones(3)

    end
    # Load ICs
    # Position
    u0_temp[1+4*NUM_DRONES] = 100*ones(3)

    # Velocity
    u0_temp[2+4*NUM_DRONES] = 100*ones(3)

    # Orientation
    u0_temp[3+4*NUM_DRONES] = 100*ones(3)

    # Angular velocity
    u0_temp[4+4*NUM_DRONES] = 100*ones(3)

    # Convert to ArrayPartition TODO: Just start with ArrayPartition directly
    u0 = ArrayPartition(u0_temp[1], u0_temp[2], u0_temp[3], u0_temp[4], u0_temp[5], u0_temp[6], 
        u0_temp[7], u0_temp[8], u0_temp[9], u0_temp[10], u0_temp[11], u0_temp[12], u0_temp[13], u0_temp[14], u0_temp[15], u0_temp[16])

    ## Setup parameters
    # Cable tension NNs - take in flattened vector inputs, output tension vector at drone and load respectively
    input_dim = 9 # Position, velocity and acceleration vectors for each drone relative to the attachment point of their attached cables on the load
    T_drone_nn = Chain(Dense(input_dim, 3, tanh)) #Chain(Dense(input_dim, 32, tanh), Dense(32, 3)) # TODO: Could try 2 hidden layers!!
    #params_sense = DroneSwarmParamsSense(T_drone_nn)

    # Initialise parameter struct
    # Note that cache values are initialised at corresponding u0 values
    j_drone = [2.32 0 0; 0 2.32 0; 0 0 4]
    p_nn_T_drone, re_nn_T_drone = Flux.destructure(T_drone_nn)

    drone_swarm_params = DroneSwarmParams_init(num_drones=NUM_DRONES, g=9.81, m_load=0.225, m_drones=[0.5, 0.5, 0.5], m_cables=[0.1, 0.1, 0.1], l_cables=[1.0, 1.0, 1.0],
                                    j_load = [2.1 0 0; 0 1.87 0; 0 0 3.97], j_drones= [j_drone, j_drone, j_drone], 
                                    r_cables = [[-0.42, -0.27, 0], [0.48, -0.27, 0], [-0.06, 0.55, 0]], re_nn_T_drone=re_nn_T_drone, 
                                    t_prev=0.0, ẋₗ_prev=u0.x[2+4*NUM_DRONES], Ωₗ_prev=u0.x[4+4*NUM_DRONES], ẋᵢ_prev=collect(u0.x[NUM_DRONES+1:2*NUM_DRONES]), 
                                    ẍₗ_prev=[0.0, 0.0, 0.0], αₗ_prev=[0.0, 0.0, 0.0], ẍᵢ_prev=Vector{Vector{Float64}}([zeros(3) for _ in 1:NUM_DRONES]))

end

# GENERATE TRAINING DATA
begin
   
    ## Generate training data
    t_step_data = 0.1
    t_data, data, ẍᵢ, αᵢ, ẍₗ, αₗ = generate_tension_training_data(t_step_data, drone_swarm_params) #t_data, xᵢ, ẋᵢ, ẍᵢ, θᵢ, Ωᵢ, αᵢ, xₗ, ẋₗ, ẍₗ, θₗ, Ωₗ, αₗ
    println()
    #println(data[end])

    # Display trajectory - load
    # xₗ = data[:][13]
    # ẋₗ = data[:][14]

    # print(xₗ)

    #plot_trajectory(t_data, xₗ, ẋₗ, ẍₗ, true, false) # TODO: Modify plotting functions to take ArrayPartitions or convert back from AP


    # plot_trajectory(t_data, xₗ, ẋₗ, ẍₗ, true, false)
    # plot_trajectory(t_data, θₗ, Ωₗ, αₗ, true, true)

    # Display trajectory - drone
    # plot_trajectory(t_data[3:end], xᵢ[3:end], ẋᵢ[3:end], ẍᵢ[3:end], false, false)
    # plot_trajectory(t_data, θᵢ[3:end], Ωᵢ[3:end], αᵢ[3:end], false, true)

end

# TRIM TRAINING DATA
begin
    # Trim training data to remove early numerical estimation errors
    step_first = 5 #4 should be enough as accel stable here and velocity stable at 2. Only need to reach back for veloctiy
    step_last = step_first+3 # TODO: REMOVE END TRIMMING (set step_last = end)
    
    t_data_trimmed = t_data[step_first:step_last]
    data_trimmed = data[step_first:step_last]

end

# Modify IC's based on training data # TODO: PUT TRAINING DATA BEFORE ICS SO DON'T HAVE TO MODIFY ICs
begin
    # Set IC's to be at step 3 from training data - where data stabilizes
    ap_u0_num_partitions = 16 #TODO: Find method to get number of partitions in array partitions rather than hard-coding

    # Print u0 to check ICs
    # println("u0 before update")
    # println(u0)
    # println()

    # Loop through all array partition components and store data at selected step in u0
    for i in 1:ap_u0_num_partitions 
        u0.x[i][1:length(u0.x[i])] = data_trimmed[1].x[i] 
    end

    # Print u0 to check ICs
    # println("u0 after update")
    # println(u0)

    # Update params that depend on ICs. Set prev values to initial values
    # TODO: TEST! These values currently matter for initial accel calculations. Switch back to u0 values if they don't
    # Note not copying as data won't change after generated
    drone_swarm_params.t_prev = t_data[step_first-1] #t_data_trimmed[1]
    
    drone_swarm_params.ẋₗ_prev = data[step_first-1].x[2+4*NUM_DRONES] #u0.x[2+4*NUM_DRONES] 
    drone_swarm_params.Ωₗ_prev = data[step_first-1].x[4+4*NUM_DRONES] #u0.x[4+4*NUM_DRONES]
    drone_swarm_params.ẋᵢ_prev = collect(data[step_first-1].x[NUM_DRONES+1:2*NUM_DRONES]) #collect(u0.x[NUM_DRONES+1:2*NUM_DRONES])
    
    drone_swarm_params.ẍₗ_prev = ẍₗ[step_first-1]
    drone_swarm_params.αₗ_prev = αₗ[step_first-1]
    drone_swarm_params.ẍᵢ_prev = ẍᵢ[step_first-1]
    
    #println(drone_swarm_params)

end


# TEST: ONE ODE CALL
begin
    # du_temp = [Vector{Float64}(undef, 3) for i in 1:(4*drone_swarm_params.num_drones+4)]
    # du = ArrayPartition(du_temp[1],du_temp[2], du_temp[3], du_temp[4], 
    #     du_temp[5], du_temp[6], du_temp[7], du_temp[8], du_temp[9], du_temp[10],du_temp[11], du_temp[12], 
    #     du_temp[13],du_temp[14], du_temp[15], du_temp[16]) # Must be a better way of initializing. This doesn't work anyway -> probably need to do du_temp[a][1:length(du_temp[a])] trick

    du = ArrayPartition([2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], 
        [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], 
        [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]) 

    t = t_data_trimmed[1] #t_data[step_u0] # Test solve at initial datapoint

    # ## Test one function call
    #println("before: $u0") #$du")
    drone_swarm_params(du, u0, p_nn_T_drone, t) # params_sense
    println()
    println("du after: $du") #$du")
end


# TEST: DOES LOSS FXN CHANGE WITH CHANGES IN NN_PARMS?
begin
    # Run ODE solver (define better u0's, proper fi's and mi's !)
    #step_size_callback = DiscreteCallback(true, print_step_size) # TODO: Tune this (when save etc.)

    # Create smaller data series for training (3 timesteps)


    l1 = loss(data_trimmed, t_data_trimmed, drone_swarm_params, u0, p_nn_T_drone)

    # p_nn_T_drone_new = copy(p_nn_T_drone)
    # p_nn_T_drone_new[end] = 10.0

    #l2 = loss(data, t_data, drone_swarm_params, u0, p_nn_T_drone_new)

    # println(p_nn_T_drone)
    # println()
    # println(p_nn_T_drone_new)
    
    println(l1) # Likely a problem with DroneSwarmParams. Result should change 
    #println(l2) # (perhaps one value is dominating the loss - try initialising smaller)

    # HELP HEREE - FD works but gives 0 sensitivity to first few?? Forward diff doesn't work??
    #typeof(p_nn_T_drone)
    # du = similar(u0)
    # println(du.x[1])

end





# TEST: Gradient
begin
    #grad = FiniteDiff.finite_difference_gradient(p_nn_T_drone-> loss(data, t_data, drone_swarm_params, u0, p_nn_T_drone), p_nn_T_drone) # Easier
    #grad2 = ForwardDiff.gradient(p_nn_T_drone-> loss(data, t_data, drone_swarm_params, u0, p_nn_T_drone), p_nn_T_drone) # Use flattened params p here. Perhaps Flux doesn't work with FwdDiff???
    #Zygote.gradient

    # t=0.0
    # out = zeros(length(p_nn_T_drone),length(u0))
    # cache = FiniteDiff.JacobianCache(p_nn_T_drone)

    # jac = FiniteDiff.finite_difference_jacobian!(du, (du, p_nn_T_drone)->(drone_swarm_params(du,u0,p_nn_T_drone,t)), p_nn_T_drone, cache)
end


# TEST: Training
begin
    ## Train
    # Train with same ODE simply with tension vectors defined using quasi-static assumption like in paper
    # Will later do using real data from simulator
    # Note only trained with specific Lambda used above. Could change lambda to train with different data 

    # optimize the parameters for a few epochs with ADAM on time span Nint
    #lr = 0.01
    # epochs = 400

    # opt = ADAM(lr)
    # list_plots = []
    # losses = []

    ## Setup and run the optimization - USE THIS NEXT!!!!
    # adtype = Optimization.AutoFiniteDiff() #.AutoZygote() # Could try "AutoForward" or someting else - Finite?? What am I doing with inputs x?? should they be p??
    # optf = Optimization.OptimizationFunction((p_nn_T_drone, p)-> loss(data, t_data, drone_swarm_params, u0, p_nn_T_drone), adtype) #(x, p) -> loss(x), adtype)

    # optprob = Optimization.OptimizationProblem(optf, p_nn_T_drone)
    # res = Optimization.solve(optprob, OptimizationOptimisers.Adam(lr), maxiters = 100) #callback = visualization_callback,                      
    
    # Make sure that function learn is easily -> Try just setting first element to T_NN to train just this.
    # Rember to add hidden layer back in T_nn_drone


    # Do I need to do data batches, epochs etc?? What updates parameters?
    # How to visualize

    # plot optimized control - not required
    #visualization_callback(res.u, loss(res.u); doplot = true)



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
# Functions to generate training data for the UDE
# 
# Author: Harvey Merton

begin
    # include("utils.jl")
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


# Generate training data using differential flatness 
function generate_training_data(t_step, params)
    # Use a circlular trajectory for the load with constant scalar velocity at walking pace 1.2 m/s
    # Will change with selected load trajectory
    v_target = 5.0 #1.2
    r_scalar = 2.0 # Radius of circular trajectory

    ω_scalar = v_target/r_scalar
    ω = [0.0, 0.0, ω_scalar]

    # Set time to capture 1 full rotation
    T = 2*π/ω_scalar 
    t_data = 0.0:t_step:T

    # Vector in z direction
    e₃ = [0,0,1] 

    ## Generate the circular load trajectory
    # Preallocate arrays
    # Drones
    xᵢ = [[Vector{Float64}(undef, 3) for _ in 1:params.num_drones] for _ in 1:length(t_data)] 
    ẋᵢ = [[Vector{Float64}(undef, 3) for _ in 1:params.num_drones] for _ in 1:length(t_data)] 
    ẍᵢ = [[Vector{Float64}(undef, 3) for _ in 1:params.num_drones] for _ in 1:length(t_data)]

    θᵢ = [[Vector{Float64}(undef, 3) for _ in 1:params.num_drones] for _ in 1:length(t_data)] 
    Ωᵢ = [[Vector{Float64}(undef, 3) for _ in 1:params.num_drones] for _ in 1:length(t_data)] 
    αᵢ = [[Vector{Float64}(undef, 3) for _ in 1:params.num_drones] for _ in 1:length(t_data)]

    # Drone inputs
    fₘ = [[[0.0, Vector{Float64}(undef, 3)] for _ in 1:params.num_drones] for _ in 1:length(t_data)]
    #print(typeof(fₘ))


    # Load
    xₗ = [Vector{Float64}(undef, 3) for _ in 1:length(t_data)]
    ẋₗ = [Vector{Float64}(undef, 3) for _ in 1:length(t_data)]
    ẍₗ = [Vector{Float64}(undef, 3) for _ in 1:length(t_data)]

    θₗ = [Vector{Float64}(undef, 3) for _ in 1:length(t_data)]
    Ωₗ = [Vector{Float64}(undef, 3) for _ in 1:length(t_data)]
    αₗ = [Vector{Float64}(undef, 3) for _ in 1:length(t_data)]

    # Tension
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
            # Note: currently tension calc is only used to get position of drone relative to load
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
            Ωᵢ[ind][i] = (θᵢ[ind][i] - θᵢ_prev[i])/t_step # TODO: SHIFT BACK 1 STEP OR USE CENTRAL DIFFERENCE

            # Angular acceleration
            αᵢ[ind][i] = (Ωᵢ[ind][i] - Ωᵢ_prev[i])/t_step # TODO: SHIFT BACK 2 STEPS OR USE CENTRAL DIFFERENCE


            # Required force
            Rᵢ = rpy_to_R(θᵢ[ind][i])
        
            v_rhs = params.m_drones[i]*ẍᵢ[ind][i] + params.m_drones[i]*params.g*e₃ - Rₗ*T₍ind₎[i]
            v_lhs = Rᵢ*e₃
            fₘ[ind][i][1] = (v_rhs./v_lhs)[3] # Taking last component works for no drone orientation change case. TODO: TEST WITH OTHER ORIENTATIONS

            # Required moment
            mᵢ = params.j_drones[i]*αᵢ[ind][i] + cross(Ωᵢ[ind][i],(params.j_drones[i]*Ωᵢ[ind][i]))
            fₘ[ind][i][2][1:end] = mᵢ[1:end]

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

    data = [deepcopy(ap_temp) for _ in 1:length(t_data)]

    # Assign elements of data array
    # Across all timesteps
    for i in 1:length(t_data)
        # Across all components
        data[i] = ArrayPartition(xᵢ[i][1], xᵢ[i][2], xᵢ[i][3], ẋᵢ[i][1], ẋᵢ[i][2], ẋᵢ[i][3], 
                                θᵢ[i][1], θᵢ[i][2], θᵢ[i][3], Ωᵢ[i][1], Ωᵢ[i][2], Ωᵢ[i][3], 
                                xₗ[i], ẋₗ[i], θₗ[i], Ωₗ[i])
        #data[i].x[1][1:length(arr_parts[1].x[1])] = [2.0, 2.0, 2.0]
    end

    return t_data, data, ẍᵢ, αᵢ, ẍₗ, αₗ, fₘ

end
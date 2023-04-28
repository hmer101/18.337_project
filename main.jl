# 18.337 Project - Learning cable tension vectors in multi-drone slung load carrying
# 
# Author: Harvey Merton


begin
   using DifferentialEquations 
   using Rotations
   using LinearAlgebra
   using Flux
   include("utils.jl")
   include("datatypes.jl")
end

begin
    const NUM_DRONES = 3 
end

# Function that returns function to give num_drones drone inputs (forces and moments) at a specified time
# Currently set values at quasi-static state-regulation inputs. TODO: might need to change to include horizonal components 
# TODO: Perhaps try trajectory-following
function drone_forces_and_moments(num_drones::Int, params, t::Float64) #, f::Vector{Float64}, m::Matrix{Float64})
    # Preallocate the output vector
    fₘ = Vector{Float64}(undef, 4 * num_drones)
    
    # Store the force inputs for each drone
    for i in 1:num_drones
        fₘ[i] = (params.m_drones[i]+params.m_cables)*params.g + params.m_load/params.num_drones
        fₘ[num_drones + 3*i - 2 : num_drones + 3*i] = [0.0;0.0;0.0] #m[:, i]
    end

    return fₘ
end


# u = x = []
# TODO: might need to flatten and unflatten matricies if using ForwardDiff
# TODO: replace all du= with du.=
function ode_sys_drone_swarm_nn!(du,u,p,t)
    # Get force and moment inputs from drones
    fₘ = drone_forces_and_moments(p.num_drones, params, t)

    ## Variable unpacking 
    e₃ = [0,0,1] 

    # Load
    xₗ = u[p.num_drones*4+1]
    ẋₗ = u[p.num_drones*4+2]
    θₗ = u[p.num_drones*4+3]
    Ωₗ = u[p.num_drones*4+4]

    Rₗ = RotZYX(θₗ[1], θₗ[2], θₗ[3]) # RPY angles to rotation matrix. Use Rotations.params(RotZYX(R)) to go other way

    ## Equations of motion
    ## Load
    ∑RₗTᵢ_load = zeros(3)
    ∑rᵢxTᵢ_load = zeros(3)

    # Calculate cumulative effect of cables on load
    for i in 1:p.num_drones
        #Tᵢ_drone = u[4*p.num_drones + 4 + i]
        Tᵢ_load = u[5*p.num_drones + 4 + i]

        # Sum across all cables needed for load EOM calculations
        ∑RₗTᵢ_load -= Rₗ*Tᵢ_load # Forces
        ∑rᵢxTᵢ_load += cross(p.r_cables[i],-Tᵢ_load) # Moments

    end

    # Load EOM
    # Velocity
    du[1+4*p.num_drones] = ẋₗ

    # Acceleration
    ẍₗ = (1/p.m_load)*(∑RₗTᵢ_load -p.m_load*g*e₃)
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

        Rᵢ = RotZYX(θᵢ[1], θᵢ[2], θᵢ[3]) # RPY angles to rotation matrix. Use Rotations.params(RotZYX(R)) to go other way

        # Connections (after drone and load in u)
        Tᵢ_drone = u[4*p.num_drones + 4 + i]
        #Tᵢ_load = u[5*p.num_drones + 4 + i]

        # Inputs 
        fᵢ = fₘ[i]
        mᵢ = fₘ[p.num_drones+i]


        ### Equations of motion
        ## Drones
        # Velocity
        du[i] = ẋᵢ

        # Acceleration
        ẍᵢ = (1/p.m_drones[i])*(fᵢ*Rᵢ*e₃ - p.m_drones[i]*p.g*e₃ + Tᵢ_drone) #ORIENTATION NOT DEFINED BY R_L??? R_L*T_i_drone)
        du[i+p.num_drones] = ẍᵢ

        # Angular velocity
        du[i+2*p.num_drones] = Ωᵢ

        # Angular acceleration
        # αᵢ = inv(p.j_drones[i])*(mᵢ - cross(Ωᵢ,(p.j_drones[i]*Ωᵢ)))
        du[i+3*p.num_drones] = inv(p.j_drones[i])*(mᵢ - cross(Ωᵢ,(p.j_drones[i]*Ωᵢ)))

        ## Connection (note these come after load indicies to make it easier to change if required)    
        # Drone motion relative to associated cable's connection point on load (for tension vector neural network)
        x_Dᵢ_rel_Lᵢ = xᵢ - (xₗ + p.r_cables[i])
        ẋ_Dᵢ_rel_Lᵢ = ẋᵢ - (ẋₗ + cross(Ωₗ,p.r_cables[i]))

        ẍ_Lᵢ = ẍₗ + cross(αₗ, p.r_cables[i]) + cross(Ωₗ, cross(Ωₗ, p.r_cables[i])) # Acceleration of point on load where cable is attached
        ẍ_Dᵢ_rel_Lᵢ = ẍᵢ - ẍ_Lᵢ - cross(αₗ, x_Dᵢ_rel_Lᵢ) - cross(Ωₗ,cross(Ωₗ,x_Dᵢ_rel_Lᵢ)) - 2*cross(Ωₗ,ẋ_Dᵢ_rel_Lᵢ)
        
        # Drone side
        du[i+4*p.num_drones+4] = p.T_dot_drone_nn(x_Dᵢ_rel_Lᵢ, ẋ_Dᵢ_rel_Lᵢ, ẍ_Dᵢ_rel_Lᵢ)

        # Load side
        du[i+4*p.num_drones+5] = p.T_dot_load_nn(x_Dᵢ_rel_Lᵢ, ẋ_Dᵢ_rel_Lᵢ, ẍ_Dᵢ_rel_Lᵢ)

    end

end

## STEP 1 - solve the ODE forwards
# Solve neural ODE
# function solve_nn_ode_fwd(nn_ode, u0, t_span)
#     # Define neural ODE
#     # Potential alternative?: NeuralODE(nn, tspan, Tsit5(), saveat = 0.1, reltol = 1e-3, abstol = 1e-3)
#     function ode_f!(du,u,p,t)
#         nn = p[1]
#         du .= nn(u)
#     end

#     prob = ODEProblem(ode_f!, u0, t_span, [nn_ode])
#     sol = solve(prob, Tsit5(), abstol = 1e-12, reltol = 1e-12)

#     return sol
# end



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
        u0[4*NUM_DRONES + 4 + i] = i*ones(3)

        # Load side
        u0[5*NUM_DRONES + 4 + i] = i*ones(3)

    end
    # Load ICs
    # Velocity
    u0[1+4*NUM_DRONES] = 100*ones(3)

    # Acceleration
    u0[2+4*NUM_DRONES] = 100*ones(3)

    # Angular velocity
    u0[3+4*NUM_DRONES] = 100*ones(3)

    # Angular acceleration
    u0[4+4*NUM_DRONES] = 100*ones(3)

    ## Setup parameters
    # Cable tension NNs - take in flattened vector inputs, output tension vector at drone and load respectively
    input_dim = NUM_DRONES*9 # Position, velocity and acceleration vectors for each drone relative to the attachment point of their attached cables on the load
    nn_T_dot_drone = Chain(Dense(input_dim, 32, tanh), Dense(32, 3)) # TODO: Currently 1 hidden layer - could try 2!!
    nn_T_dot_load = Chain(Dense(input_dim, 32, tanh), Dense(32, 3))

    # Initialise parameter struct
    j_drone = [2.32 0 0; 0 2.32 0; 0 0 4]

    params = DroneSwarmParams_init(num_drones=NUM_DRONES, g=9.81, m_load=0.225, m_drones=[0.5, 0.5, 0.5], m_cables=[0.1, 0.1, 0.1], 
                                    j_load = [2.1 0 0; 0 1.87 0; 0 0 3.97], j_drones= [j_drone, j_drone, j_drone], 
                                    r_cables = [[-0.42, -0.27, 0], [0.48, -0.27, 0], [-0.06, 0.55, 0]], T_dot_drone_nn=nn_T_dot_drone, T_dot_load_nn=nn_T_dot_load)

end

begin
    # TEST!!!!!!!!!
    ## Solve
    du = 1
    t = 1

    ode_sys_drone_swarm_nn!(du,u,p,t)


    # # Example parameter
    # a = 0.5

    # # Example initial conditions and time span (assuming u contains 2x2 matrices)
    # u0 = [rand(2, 2) for _ in 1:3]
    # tspan = (0.0, 1.0)

    # # Solve the ODE system
    # using DifferentialEquations
    # prob = ODEProblem(ode_sys_drone_swarm_nn!, u0, tspan, (nn_model1, nn_model2, a))
    # sol = solve(prob, Tsit5()) 



    ## Train
    # Train with same ODE simply with tension vectors defined using quasi-static assumption like in paper
    # Will later do using real data from simulator

end
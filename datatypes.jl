# File that contains custom data types
# 
# Author: Harvey Merton

# Define the custom struct
mutable struct DroneSwarmParams
    # Constants
    g::Float64
    
    # Parameters defining swarm
    num_drones::Int64

    m_load::Float64
    m_drones::Vector{Float64}
    m_cables::Vector{Float64}
    l_cables::Vector{Float64}

    j_load::Matrix{Float64}
    j_drones::Vector{Matrix{Float64}}

    r_cables::Vector{Vector{Float64}}

    # Parameters defining solve
    t_step::Float64 # Step size used for generating data
    t_start::Float64 # Start time used for solving ODE
    fₘ::Vector{Vector{Vector{Any}}} # Force and moment inputs to generate data

    # Parameters for NNs
    re_nn_T_drone::Any

    # Cache for FD approximations
    t_prev::Float64 # TODO: REMOVE t CACHE AS NO LONGER REQUIRED FOR FD

    ẋₗ_prev::Vector{Float64} # TODO: REMOVE VELOCITY CACHE AS NO LONGER REQUIRED FOR FD
    Ωₗ_prev::Vector{Float64}
    ẋᵢ_prev::Vector{Vector{Float64}}

    ẍₗ_prev::Vector{Float64} #_est
    αₗ_prev::Vector{Float64} #_est
    ẍᵢ_prev::Vector{Vector{Float64}} #_est

    # History of tensions and predicted relative velocities
    x₍i_rel_Lᵢ₎_hist::Vector{Vector{Vector{Float64}}}
    ẋ₍i_rel_Lᵢ₎_hist::Vector{Vector{Vector{Float64}}}
    ẍ₍i_rel_Lᵢ₎_hist::Vector{Vector{Vector{Float64}}}
    
    Tᵢ_drone_hist::Vector{Vector{Vector{Float64}}}
    
end

# Custom struct to hold parameters that we want the sensitivity to
# mutable struct DroneSwarmParamsSense
#     T_drone_nn::Chain
#     #T_dot_load_nn::Chain
# end



function DroneSwarmParams_init(; g::Float64, num_drones::Int64, m_load::Float64, m_drones::Vector{Float64}, m_cables::Vector{Float64}, l_cables::Vector{Float64},
    j_load::Matrix{Float64}, j_drones::Vector{Matrix{Float64}}, r_cables::Vector{Vector{Float64}}, t_step::Float64, t_start::Float64, fₘ::Vector{Vector{Vector{Any}}}, re_nn_T_drone::Any, 
    t_prev::Float64, ẋₗ_prev::Vector{Float64}, Ωₗ_prev::Vector{Float64}, ẋᵢ_prev::Vector{Vector{Float64}}, ẍₗ_prev::Vector{Float64}, αₗ_prev::Vector{Float64}, ẍᵢ_prev::Vector{Vector{Float64}},
    x₍i_rel_Lᵢ₎_hist::Vector{Vector{Vector{Float64}}}, ẋ₍i_rel_Lᵢ₎_hist::Vector{Vector{Vector{Float64}}}, ẍ₍i_rel_Lᵢ₎_hist::Vector{Vector{Vector{Float64}}}, Tᵢ_drone_hist::Vector{Vector{Vector{Float64}}})

    # Use deepcopy to allow cache to be updated without affecting original data
    # Don't need to deepcopy variables that won't change, but safer to do so
    return DroneSwarmParams(deepcopy(g), deepcopy(num_drones), deepcopy(m_load), deepcopy(m_drones), deepcopy(m_cables), 
                            deepcopy(l_cables), deepcopy(j_load), deepcopy(j_drones), deepcopy(r_cables), 
                            deepcopy(t_step), deepcopy(t_start), deepcopy(fₘ), deepcopy(re_nn_T_drone), deepcopy(t_prev), 
                            deepcopy(ẋₗ_prev), deepcopy(Ωₗ_prev), deepcopy(ẋᵢ_prev), 
                            deepcopy(ẍₗ_prev), deepcopy(αₗ_prev), deepcopy(ẍᵢ_prev),
                            deepcopy(x₍i_rel_Lᵢ₎_hist), deepcopy(ẋ₍i_rel_Lᵢ₎_hist), deepcopy(ẍ₍i_rel_Lᵢ₎_hist), deepcopy(Tᵢ_drone_hist))
end


mutable struct TrainingData
    iter_cnt::Int
    L_hist::Vector{Float64}

end
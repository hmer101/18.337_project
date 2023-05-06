# File that contains custom data types
# 
# Author: Harvey Merton

# using DifferentialEquations
# using NeuralPDE
using Flux

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
    #t_step::Float64

    # Parameters for NNs
    re_nn_T_drone::Any

    # Cache for FD approximations
    t_prev::Float64

    ẋₗ_prev::Vector{Float64}
    Ωₗ_prev::Vector{Float64}
    ẋᵢ_prev::Vector{Vector{Float64}}
    
end

# Custom struct to hold parameters that we want the sensitivity to
mutable struct DroneSwarmParamsSense
    T_drone_nn::Chain
    #T_dot_load_nn::Chain
end



function DroneSwarmParams_init(; g::Float64, num_drones::Int64, m_load::Float64, m_drones::Vector{Float64}, m_cables::Vector{Float64}, l_cables::Vector{Float64},
    j_load::Matrix{Float64}, j_drones::Vector{Matrix{Float64}}, r_cables::Vector{Vector{Float64}}, re_nn_T_drone::Any, t_prev::Float64, ẋₗ_prev::Vector{Float64}, Ωₗ_prev::Vector{Float64}, ẋᵢ_prev::Vector{Vector{Float64}})

    return DroneSwarmParams(g, num_drones, m_load, m_drones, m_cables, l_cables, j_load, j_drones, r_cables, re_nn_T_drone, t_prev, ẋₗ_prev, Ωₗ_prev, ẋᵢ_prev)
end


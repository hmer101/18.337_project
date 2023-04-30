# File that contains custom data types
# 
# Author: Harvey Merton

# using DifferentialEquations
# using NeuralPDE
# using Flux

# # Define the Neural ODE struct
# mutable struct NeuralODE
#     input_dim::Int
#     output_dim::Int
#     model::Chain
#     p::Vector{Float32}
# end

# Define the custom struct
mutable struct DroneSwarmParams
    num_drones::Int64
    g::Float64

    m_load::Float64
    m_drones::Vector{Float64}
    m_cables::Vector{Float64}

    j_load::Matrix{Float64}
    j_drones::Vector{Matrix{Float64}}

    r_cables::Vector{Vector{Float64}}
    
    #use_nn::Bool
    T_dot_drone_nn::Chain
    T_dot_load_nn::Chain
end
#num_drones, g, m_drones, m_l, nn


function DroneSwarmParams_init(; num_drones::Int64, g::Float64, m_load::Float64, m_drones::Vector{Float64}, m_cables::Vector{Float64}, 
    j_load::Matrix{Float64}, j_drones::Vector{Matrix{Float64}}, r_cables::Vector{Vector{Float64}}, T_dot_drone_nn::Chain, T_dot_load_nn::Chain)

    return DroneSwarmParams(num_drones, g, m_load, m_drones, m_cables, j_load, j_drones, r_cables, T_dot_drone_nn, T_dot_load_nn)
end


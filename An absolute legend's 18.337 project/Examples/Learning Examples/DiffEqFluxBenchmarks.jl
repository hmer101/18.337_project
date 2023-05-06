using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots, DiffEqSensitivity,
      Zygote, BenchmarkTools, Random

u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

dudt2 = FastChain((x, p) -> x.^3,
                  FastDense(2, 50, tanh),
                  FastDense(50, 2))
Random.seed!(100)
p = initial_params(dudt2)

prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

function loss_neuralode(p)
    pred = Array(prob_neuralode(u0, p))
    loss = sum(abs2, ode_data .- pred)
    return loss
end

@btime Zygote.gradient(loss_neuralode,p)
# 2.709 ms (56506 allocations: 6.62 MiB)

prob_neuralode_interpolating = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))

function loss_neuralode_interpolating(p)
    pred = Array(prob_neuralode_interpolating(u0, p))
    loss = sum(abs2, ode_data .- pred)
    return loss
end

@btime Zygote.gradient(loss_neuralode_interpolating,p)
# 5.501 ms (103835 allocations: 2.57 MiB)

prob_neuralode_interpolating_zygote = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))

function loss_neuralode_interpolating_zygote(p)
    pred = Array(prob_neuralode_interpolating_zygote(u0, p))
    loss = sum(abs2, ode_data .- pred)
    return loss
end

@btime Zygote.gradient(loss_neuralode_interpolating_zygote,p)
# 2.899 ms (56150 allocations: 6.61 MiB)

prob_neuralode_backsolve = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps, sensealg=BacksolveAdjoint(autojacvec=ReverseDiffVJP(true)))

function loss_neuralode_backsolve(p)
    pred = Array(prob_neuralode_backsolve(u0, p))
    loss = sum(abs2, ode_data .- pred)
    return loss
end

@btime Zygote.gradient(loss_neuralode_backsolve,p)
# 4.871 ms (85855 allocations: 2.20 MiB)

prob_neuralode_quad = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)))

function loss_neuralode_quad(p)
    pred = Array(prob_neuralode_quad(u0, p))
    loss = sum(abs2, ode_data .- pred)
    return loss
end

@btime Zygote.gradient(loss_neuralode_quad,p)
# 11.748 ms (79549 allocations: 3.87 MiB)

prob_neuralode_backsolve_tracker = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps, sensealg=BacksolveAdjoint(autojacvec=TrackerVJP()))

function loss_neuralode_backsolve_tracker(p)
    pred = Array(prob_neuralode_backsolve_tracker(u0, p))
    loss = sum(abs2, ode_data .- pred)
    return loss
end

@btime Zygote.gradient(loss_neuralode_backsolve_tracker,p)
# 27.604 ms (186143 allocations: 12.22 MiB)

prob_neuralode_backsolve_zygote = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps, sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP()))

function loss_neuralode_backsolve_zygote(p)
    pred = Array(prob_neuralode_backsolve_zygote(u0, p))
    loss = sum(abs2, ode_data .- pred)
    return loss
end

@btime Zygote.gradient(loss_neuralode_backsolve_zygote,p)
# 2.091 ms (49883 allocations: 6.28 MiB)

prob_neuralode_backsolve_false = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps, sensealg=BacksolveAdjoint(autojacvec=ReverseDiffVJP(false)))

function loss_neuralode_backsolve_false(p)
    pred = Array(prob_neuralode_backsolve_false(u0, p))
    loss = sum(abs2, ode_data .- pred)
    return loss
end

@btime Zygote.gradient(loss_neuralode_backsolve_false,p)
# 4.822 ms (9956 allocations: 1.03 MiB)

prob_neuralode_tracker = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps, sensealg=TrackerAdjoint())

function loss_neuralode_tracker(p)
    pred = Array(prob_neuralode_tracker(u0, p))
    loss = sum(abs2, ode_data .- pred)
    return loss
end

@btime Zygote.gradient(loss_neuralode_tracker,p)
# 12.614 ms (76346 allocations: 3.12 MiB)

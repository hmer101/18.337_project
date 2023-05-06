## This is a script intended to house benchmarking code that would not otherwise be useful at runtime. (It is not imported into the main module.)

using BenchmarkTools
using Profile

## Set up data

# TODO need to define
    # u0
    # d
    # t_d

# To make sure all the benchmark data is consistent, use this relatively small neural network
NN_benchmark = Chain(
    Dense(4, 16, tanh),
    Dense(16, 16, tanh),
    Dense(16, 4)
)
# These variables will be needed by some of the following naive functions
p, re = Flux.destructure(NN_benchmark) # Gets p as a vector; re is the function to restructure the NN
θ = [u0; p]


## How to properly use a neural network (why to use function generators)

# Naive neural network wrapper TODO needs to be fed re(), after neural network is made
function f_1(u, p, t)
    re(p)(u) # Reconstruct the NN, and evaluate it
end

# What if we only evaluated re() once? (It's expensive!)
rec = re(p)

# Two different ways to use that
function f_2(u, p, t)
    rec(u)
end
f_3(u, p, t) = rec(u) # Inline version

# Diagnose problems with f_1
@btime f_1(u0, p, 0) # 6.5 microseconds, 81 allocations, 5.66 KiB
@btime f_1(θ[1:4], θ[5:end], 0) # 7.4 microseconds, 83 allocations, 11.11 KiB
@btime f_1(@view(θ[1:4]), @view(θ[5:end]), 0) # 7.4 microseconds, 91 allocations, 8.28 KiB
@code_warntype f_1(u0, p, 0) # Returns Any! Takes in things of known type.

# Identify cost of re()
@btime re(p) # 5.6 microseconds, 74 allocations
@code_warntype re(p) # Takes in known type, returns Any
@code_warntype rec(u0) # Type-stable; so it's just the creation step that's type-unstable

# Try the alternatives to f_1() and check their cost
@btime f_2(u0, p, 0) # 758 nanoseconds, 7 allocations, 864 bytes
@btime f_3(u0, p, 0) # 760 nanoseconds, 7 allocations, 864 bytes
@btime NN_1(u0) # 731 nanoseconds (7 allocations, 864 bytes)

# The conclusion from all this should be pretty clear: f_2 is faster than f_1. But it doesn't seem to matter (and might even be slightly worse) when going to functions that actually depend on it...


## Testing forward calculation speed of FastChain

# Try the FastChain type instead
NN_benchmark_2 = FastChain(
    FastDense(4, 16, tanh),
    FastDense(16, 16, tanh),
    FastDense(16, 4)
)

pp = initial_params(NN_benchmark_2)
@btime NN_2(u0, pp) # 1.38 microseconds (24 allocations) - so this isn't actually better than Flux's neural networks...


## Validating that the smoothing indeed works (this is more of a test)

# TODO All of these variables need to be defined...maybe I should just load in some dummy data at the top of this file, can be anything

# Verify that a trajectory using the derivative of the smoothed data is somewhat realistic (before we train a NN to mimic the derivative, better make sure it's not garbage)
using Interpolations
# Interpolate the derivative so values can be drawn from anywhere
itp = interpolate(dd_smoothed, (NoInterp(), BSpline(Linear())))
itp = scale(itp, 1:size(d, 1), range(t_d[1], t_d[end], length = length(t_d))) # Scale it against the time, rather than indices
f_1_smooth(u, p, t) = (i->itp(i, t)).(1.0:Float64(size(d, 1))) # To be used in an ODE
# Set up and solve an ODEProblem
problem_smooth = ODEProblem(f_1_smooth, d_smoothed[:, 1], tspan)
sol_smooth = solve(problem_smooth, Tsit5())
plot(sol_smooth, title = "Integration of Smoothed Derivative") # Should overlap almost exactly, the only difference being from errors in Tsit5() [which isn't much] and the fact the data is interpolated
plot!(t_d, d_smoothed')

# NOTE: This collocaiton function REALLY doesn't like my data...I have a suspicion that the only times it doesn't fail it's just spitting out the original data as a quiet failure
# Get the collocation to get the approximate "true" derivative of the data
#interp_du, interp_u = collocate_data(d_smoothed, t_d, GaussianKernel())
    # The reason this sometimes fails is that there's not enough data to get a good fit (I think, based on testing in the example script)
    # Didn't error: Gaussian, logistic, sigmoid, Silverman
    # But none of those are any good. Way too much noise, I guess; it overfits the noise. Or might just ping back what it was input.
#plot(t_d, d')
#plot(t_d, interp_u')
#plot(t_d, interp_du')


## Naive loss functions for collocation

# TODO need to define d_smoothed, dd_smoothed.......

# Separate loss function for collocation
function loss_collocation_fast(p)
    loss = zero(first(p)) # TODO This is clearly a copy-pasted line, might want to change it to e.g. specialize on my problems...
    for i in 1:size(dd_smoothed, 2)
        # Attempt to match f_1 to the interpolated derivative
        #f_1( d_smoothed[:, i], p)
        loss += sum(abs2, (@view dd_smoothed[:, i]) .- f_1( (@view d_smoothed[:, i]), p, 0) )
    end
    return loss
end

# Separate loss function for collocation
function loss_collocation_forplots(p)
    # This will be slower than the above by at least one allocation

    # Build the full predicted derivative across the domain (not necessary for loss function, but useful for diagnostics)
    #predicted_du = similar(dd_smoothed)
    predicted_du = hcat((i->f_1( (@view d_smoothed[:, i]), p, 0)).(1:size(dd_smoothed, 2))...) # Note that f_1 does not use t, but still needs the placeholder TODO sadly mutating arrays not supported

    # Attempt to match the predicted derivative to the interpolated derivative
    loss = sum(abs2, dd_smoothed .- predicted_du)

    return loss, predicted_du

end

# Maybe if you use a function generator it would be better...?
function loss_collocation_generator(d_smoothed, dd_smoothed)

    return function (p)
        predicted_du = hcat((i->f_1( (@view d_smoothed[:, i]), p, 0)).(1:size(dd_smoothed, 2))...)
        loss = sum(abs2, dd_smoothed .- predicted_du)
    end

end

function f_2_generator(re, p)

    rec = re(p)
    return function (u, p, t)
        rec(u)
    end

end

function loss_collocation_generator_2(d_smoothed, dd_smoothed, f)

    return function (p)
        s = size(dd_smoothed)
        predicted_du = Array{Float64, 2}(undef, s)
        for i in 1:s[2]
            predicted_du[:, i] .= f( d_smoothed[:, i], p, 0 ) # sciml_train() complains about the mutation here
        end
        #predicted_du = Array{Float64, 2}(hcat((i->f_1( (@view d_smoothed[:, i]), p, 0)).(1:size(dd_smoothed, 2))...))
        loss = sum(abs2, dd_smoothed .- predicted_du)
        return loss, predicted_du
    end

end

function loss_collocation_2(p)

    # Generate type-stable functions for evaluation
    generated_f_2 = f_2_generator(re, p) # 5.7 microseconds, 75 allocations (presumably all from reconstruct())
    generated_forplots_2 = loss_collocation_generator_2(d_smoothed, dd_smoothed, generated_f_2) # 88 nanoseconds, 1 allocation

    # Calculate the actual loss
    return generated_forplots_2(p)

end

function loss_collocation_generator_3(d_smoothed, dd_smoothed, f)

    return function (p)
        s = size(dd_smoothed)
        predicted_du = Array{Float64, 2}(undef, s)
        #temp = Vector{Float64}(undef, s[1])
        @inbounds for i in 1:s[2]
            #predicted_du[:, i] .= f( d_smoothed[:, i], p, 0 )
            temp = f( d_smoothed[:, i], p, 0 ) # It hurts not to use .=, but it's still better than using the fully allocating version. And aha, because there were already allocations inside f, this doesn't actually result in any more!
            for j in 1:s[1]
                predicted_du[j, i] = temp[j]
            end
        end
        #predicted_du = Array{Float64, 2}(hcat((i->f_1( (@view d_smoothed[:, i]), p, 0)).(1:size(dd_smoothed, 2))...))
        loss = sum(abs2, dd_smoothed .- predicted_du)
        return loss, predicted_du
    end

end

function loss_collocation_3(p)

        # Generate type-stable functions for evaluation
        generated_f_2 = f_2_generator(re, p) # 5.7 microseconds, 75 allocations (presumably all from reconstruct())
        generated_forplots_3 = loss_collocation_generator_3(d_smoothed, dd_smoothed, generated_f_2) # 88 nanoseconds, 1 allocation

        # Calculate the actual loss
        return generated_forplots_3(p)

end

function loss_collocation_generator_4(d_smoothed, dd_smoothed, f)

    return function (p)
        predicted_du = hcat((i->f( (@view d_smoothed[:, i]), p, 0)).(1:size(dd_smoothed, 2))...) # Hopefully it's type stable now, so fewer allocations?
        loss = sum(abs2, dd_smoothed .- predicted_du)
        return loss, predicted_du
    end

end

function loss_collocation_4(p)

        # Generate type-stable functions for evaluation
        generated_f_2 = f_2_generator(re, p) # 5.7 microseconds, 75 allocations (presumably all from reconstruct())
        generated_forplots_4 = loss_collocation_generator_4(d_smoothed, dd_smoothed, generated_f_2) # 88 nanoseconds, 1 allocation

        # Calculate the actual loss
        return generated_forplots_4(p)

end



## Benchmarking loss functions for collocation

# Forward Calculations

@btime loss_collocation_fast(p)
@btime loss_collocation_forplots(p) # 4.8 milliseconds, 58868 allocations
# TODO : Turns out they are virtually identical because I suck at both of them and there are allocations right out the ass
    # LOL it's actually FASTER to use _forplots; I guess the compiler can do some magic in the way I wrote it out now...

Profile.clear()
@profile for i in 1:100 loss_collocation_forplots(p) end
Juno.profiler()
@code_warntype loss_collocation_forplots(p)

generated_forplots = loss_collocation_generator(d_smoothed, dd_smoothed)
@code_warntype generated_forplots(p)
@btime generated_forplots(p) # 4.6 milliseconds, 57107 allocations
@trace generated_forplots(p)

@btime generated_f_2 = f_2_generator(re, p) # 5.7 microseconds, 75 allocations (presumably all from reconstruct())
@btime generated_forplots_2 = loss_collocation_generator_2(d_smoothed, dd_smoothed, generated_f_2) # 88 nanoseconds, 1 allocation
@code_warntype generated_forplots_2(p) # TYPE STABLE AT LAST
@btime generated_forplots_2(p) # 546 microseconds, 5557 allocations - HUGE SUCCESS YAY FINALLY A VICTORY
    # That's pretty close to the minimum you could expect : 7 * 694 = 4858 allocations bare minimum. The rest probably comes from sum().
    # I still don't really get why the neural network needs to allocate anything...
Profile.clear()
@profile for i in 1:1000 generated_forplots_2(p) end
Juno.profiler()

# Putting it all together:
@btime loss_collocation_2(p) # 550 microseconds (5633 allocations) - Awesome! Works as expected now.
@btime loss_collocation_3(p) # Same (one more allocation) - A failed attempt for compatibility with Zygote
@btime loss_collocation_4(p) # 553 microseconds, 4948 allocations - Huzzah, even an improvement!
    # I guess the hcat() is okay now, since everything else is type-stable
    # And it's even closer to the ideal optimum...4858 is the minimum, so only 90 allocations overhead.
    # And it's (at last) compatible with Zygote!
# Result is about a 10x speedup.
@btime loss_collocation_5(p) # 543 microseconds, 4965 allocations - no change


# More benchmarking : the GRADIENT is what actually matters
# Note: These run faster on a fresher Julia instance (I noticed after restarting computer)

function lc0(p) # Wrapper just for testing Zygote
    out, _ = loss_collocation_forplots(p)
    return out
end

Zygote.gradient(lc0, p)
@btime Zygote.gradient(lc0, p) # 30.144 milliseconds, 245889 allocations - woah!

function lc4(p) # Wrapper just for testing Zygote
    out, _ = loss_collocation_4(p)
    return out
end

Zygote.gradient(lc4, p)
@btime Zygote.gradient(lc4, p) # 18.040 milliseconds, 149079 allocations

function lc5(p)
    out, _ = loss_collocation_5(p)
    return out
end
Zygote.gradient(lc5, p)
@btime Zygote.gradient(lc5, p) # 18.075 milliseconds, 151214 allocations

using ForwardDiff
ForwardDiff.gradient(lc4, p)
@btime ForwardDiff.gradient(lc4, p) # 110 milliseconds, 149594 allocations - okay, so Zygote is still certainly better


# Benchmarking the callbacks
# TODO This will probably be broken by the new callback interface

temp = loss_collocation_4(p)
@btime begin # Without printing anything to the REPL: 250 nanoseconds, 3 allocations. Printing one line to the REPL (using display): 609 microseconds, 113 allocations.
    # For speed, best not to print. However, each training iteration takes on the order of milliseconds, so there's no actual harm in printing.
    collocation_plot_ind[1] = 1
    callback_collocation(p, temp...)
end
@btime begin # When plotting, it takes a full 900 milliseconds, with 4,566,436 allocations (yikes! Plotting is super slow!)
    # One plot is the equivalent of 900 milliseconds / (57 milliseconds for the loss gradient calculation) = 16 training iterations
    # To keep the slowdown from plotting below, say, 10%, plot_every should be 160 iterations. At 100 iterations, it's a 16% slowdown, which is acceptable.
    collocation_plot_ind[1] = 0
    callback_collocation(p, temp...)
end


# Benchmarking the overall impact on training

result_collocation = DiffEqFlux.sciml_train(loss_collocation_forplots, p, ADAM(0.05), maxiters = 5000, cb = callback_collocation)
    # This goes at 9.0 seconds per 100 iterations, so 7.5 minutes for 5000 iterations
result_collocation = DiffEqFlux.sciml_train(loss_collocation_5, p, ADAM(0.05), maxiters = 10000, cb = callback_collocation)
    # Timing: This cranks along at 7.4 seconds per 100 iterations, so a little over 6 minutes for 5000 iterations.
    # Shoot, all that work and it's not really that much faster. I guess the key is to remember: the GRADIENT, not the function itself, is the expensive part.
    # It was *an* improvement, but not a big enough one to spend several hours on....


## Naive prediction functions for shooting method

function predict(p_, t_d)
    # This outputs a matrix the same shape as d
    Array(solve(problem, Tsit5(), u0 = p_[1:4], p = p_[4+1:end], saveat = t_d))
end

function generate_predict_2(θ, d, re, tspan)

    s = size(d, 1)
    p = θ[s+1:end]
    f = f_2_generator(re, p)

    return function(θ)
        #@show typeof(f)
        #@show typeof(u0)
        #@show typeof(tspan)
        u0 = θ[1:s]
        problem = ODEProblem(f, u0, tspan)
        Array(solve(problem, Tsit5(), p = p, saveat = t_d))
    end
end

@btime fun_predict_2 = generate_predict_2(d, re, tspan) # 88 nanoseconds, 1 allocation
@code_warntype generate_predict_2(d, re, tspan)
@btime fun_predict_2(θ) # 824 microseconds, 15156 allocations
@code_warntype fun_predict_2(θ) # It's not type stable

function predict_2(θ)
    generated_predict_2 = generate_predict_2(θ, d, re, tspan)
    return fun_predict_2(θ)
end
@btime predict_2(θ) # 892 microseconds, 15641 allocations

function generate_f_3(θ, d, re)
    p = θ[size(d, 1)+1:end]
    f = f_2_generator(re, p)
end

function generate_predict_3(f, d, tspan, t_d)
    return function(θ)
        u0 = θ[1:size(d, 1)]
        p = θ[size(d, 1)+1:end]
        problem = ODEProblem(f, u0, tspan)
        return Array(solve(problem, Tsit5(), p = p, saveat = t_d))
    end
end

f_3_generated = generate_f_3(θ, d, re)
@code_warntype generate_f_3(θ, d, re) # This returns an Any
predict_3 = generate_predict_3(f_3_generated, d, tspan, t_d)
@code_warntype generate_predict_3(f_3_generated, d, tspan, t_d) # This is now statically typed!
predict_3(θ)
@btime predict_3(θ) # 810 microseconds, 15007 allocations
@code_warntype predict_3(θ)

function generate_prob_4(θ, f, d, tspan)
    u0 = θ[1:size(d, 1)]
    return problem = ODEProblem(f, u0, tspan)
end

function predict_4(problem, t_d)
    return Array(solve(problem, Tsit5(), saveat = t_d))
end

prob_4 = generate_prob_4(θ, f_3_generated, d, tspan)
predict_4(prob_4, t_d)
@code_warntype generate_prob_4(θ, f_3_generated, d, tspan)
@code_warntype predict_4(prob_4, t_d)
@btime predict_4(prob_4, t_d) # 742 microseconds, 14662 allocations

function generate_predict_5(θ)
    f_3_generated = generate_f_3(θ, d, re)
    prob_4 = generate_prob_4(θ, f_3_generated, d, tspan)
    return function(θ)
        predict_4(prob_4, t_d)
    end
end
fun_predict_5 = generate_predict_5(θ)
@btime fun_predict_5(θ) # 821 microseconds, 15072 allocations
@code_warntype fun_predict_5(θ) # Still not type-stable...the trick isn't working

function predict_6(θ)
    f_3_generated = generate_f_3(θ, d, re)
    prob_4 = generate_prob_4(θ, f_3_generated, d, tspan)
    predict_4(prob_4, t_d)
end
@btime predict_6(θ) # 823 microseconds, 15072 allocations
@code_warntype predict_6(θ)

# And now the actual one that I'm going to stick with

@btime predict_7(θ) # 822 microseconds, 15077 alocations
@code_warntype predict_7(θ) # It ain't type stable, but it's the best I can do

@btime predict(θ, t_d) # 1.625 ms (24398 allocations)
@btime predict_2(θ, re, problem) # 1.712 ms (25556 allocations)

@code_warntype predict(θ, t_d)
predict_2(θ, re, problem)
@code_warntype predict_2(θ, re, problem)

using Profile
@profile for i in 1:1000 predict(θ, t_d) end
Juno.profiler()

# Recalling the lessons from before, let's examine the gradient timings

function l(p)
    out, _ = loss(p)
    return out
end

@btime Zygote.gradient(l, θ)
    # With predict_7 : 38.086 milliseconds, 604488 allocations
    # With predict_3 : 49.116 milliseconds, 710648 allocations (worse)
    # With predict_2 : 48.312 milliseconds, 711509 allocations (worse)
    # With predict_6 : 38.214 milliseconds, 604484 allocations
    # With predict_5 (generation inside) : 38.710 milliseconds, 604505 allocations
    # With predict_4 (generation inside) : 38.330 milliseconds, 604393 allocations
        # With the original predict() : 283.238 milliseconds, 1741360 allocations - Aha, so I have actually made a pretty meaningful improvement here! :) 7.4x faster, in the gradient

loss(θ)
Profile.clear()
@profile for i in 1:10 Zygote.gradient(l, θ) end
Juno.profiler()

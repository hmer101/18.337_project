# Simple L2 loss on an ODE
#function ODE_L2(ode_problem, d, t_d)
    # problem is an ODEProblem that shall be solved
    # d is the data we're trying to hit (a matrix, each timestamp being one column)
    # t_d are the timestamps for the data (a vector)

#    return function loss(p) # Note p is not actually used
#        u = solve(ode_problem, Tsit5())#, abstol = 1e-12, reltol = 1e-12)
#        return sum( ( i -> sum((u(t_d[i]) - d[1:end, i]).^2) ).(1:length(t_d)) ), u # TODO Surely there is a more elegant way of doing this, that Reverse-mode AD plays nicely with; see the actual ODE machine learning examples!!!!!!!!
#        # TODO also stop making it return u when doing combine_objectives, or make it handle it more gracefully
#    end

#end



# NOTE: Actually, I think this kind of usage is too simple. It's already encompassed by the actual libraries.
#function ODE_neural(NN, p)
    # Create a simple neural ODE with no time-history: du/dt = NN(u)
    # NN is a neural network defined by Flux
    # p are the parameters that will be used inside the optimization TODO figure out whether these are vectors or the fancy flux parameters

#    return function dudt(u, p, t)
        # Ignore the p parameter, since that will have been incorporated already

#    end

#end

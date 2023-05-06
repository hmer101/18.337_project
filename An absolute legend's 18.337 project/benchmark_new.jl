# As with the other script, this guy basically has to get run after all the setup in main.jl

## Automatically benchmark to figure out which adjoint is preferable

sensealg = [BacksolveAdjoint, InterpolatingAdjoint, QuadratureAdjoint]
autojacvec = [ReverseDiffVJP(true), ZygoteVJP(), TrackerVJP()] # The last one there is forward-mode

fs = []
ps = []
for i in 1:length(sensealg)
    for j in 1:length(autojacvec)

        d = deepcopy(PCA_all[i][:, 1:end])
        t_d = deepcopy(timings[i][1:end])
        f, p = benchmark_adjoint!(NN_1, d, t_d, sensealg[i], autojacvec[j])
        append!(fs, [f])
        append!(ps, [p])

    end
end

# @btime doesn't allow you to flexibly populate any of the arguments, so I'll write it all by hand...
println("Entered benchmark: " * string(sensealg[1]) * "; " * string(autojacvec[1]))
@btime fs[1](ps[1])
println("Entered benchmark: " * string(sensealg[1]) * "; " * string(autojacvec[2]))
@btime fs[2](ps[2])
println("Entered benchmark: " * string(sensealg[1]) * "; " * string(autojacvec[3]))
@btime fs[3](ps[3])
println("Entered benchmark: " * string(sensealg[2]) * "; " * string(autojacvec[1]))
@btime fs[4](ps[4])
println("Entered benchmark: " * string(sensealg[2]) * "; " * string(autojacvec[2]))
@btime fs[5](ps[5])
println("Entered benchmark: " * string(sensealg[2]) * "; " * string(autojacvec[3]))
@btime fs[6](ps[6])
println("Entered benchmark: " * string(sensealg[3]) * "; " * string(autojacvec[1]))
@btime fs[7](ps[7])
println("Entered benchmark: " * string(sensealg[3]) * "; " * string(autojacvec[2]))
@btime fs[8](ps[8])
println("Entered benchmark: " * string(sensealg[3]) * "; " * string(autojacvec[3]))
@btime fs[9](ps[9])

# Entered benchmark: BacksolveAdjoint; ReverseDiffVJP{true}()
# 644.000 μs (13251 allocations: 1.29 MiB)
# 643.699 μs (13251 allocations: 1.29 MiB)
# Entered benchmark: BacksolveAdjoint; ZygoteVJP()
# 649.200 μs (13349 allocations: 1.30 MiB)
# 651.400 μs (13349 allocations: 1.30 MiB)
# Entered benchmark: BacksolveAdjoint; TrackerVJP()
# 820.600 μs (15799 allocations: 1.55 MiB)
# 816.900 μs (15799 allocations: 1.55 MiB)

# Entered benchmark: InterpolatingAdjoint; ReverseDiffVJP{true}()
# 2.030 ms (34713 allocations: 3.50 MiB)
# 2.026 ms (34713 allocations: 3.50 MiB)
# Entered benchmark: InterpolatingAdjoint; ZygoteVJP()
# 1.062 ms (19425 allocations: 1.93 MiB)
# 1.062 ms (19425 allocations: 1.93 MiB)
# Entered benchmark: InterpolatingAdjoint; TrackerVJP()
# 880.500 μs (17073 allocations: 1.68 MiB)
# 880.401 μs (17073 allocations: 1.68 MiB)

# Entered benchmark: QuadratureAdjoint; ReverseDiffVJP{true}()
# 752.600 μs (14918 allocations: 1.46 MiB)
# 750.901 μs (14918 allocations: 1.46 MiB)
# Entered benchmark: QuadratureAdjoint; ZygoteVJP()
# 974.701 μs (18642 allocations: 1.84 MiB)
# 975.700 μs (18642 allocations: 1.84 MiB)
# Entered benchmark: QuadratureAdjoint; TrackerVJP()
# 1.464 ms (27266 allocations: 2.73 MiB)
# 1.465 ms (27266 allocations: 2.73 MiB)

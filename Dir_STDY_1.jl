using DifferentialEquations
using Plots
using Random
using LsqFit

global seed = 1234
Random.seed!(seed)

function lif(u, p, t)
    gL, EL, C, Vth, I, noise_std = p
    I_noisy = I + noise_std * randn()
    (-gL * (u - EL) + I_noisy) / C
end

function thr(u, t, integrator)
    integrator.u > integrator.p[4]
end

function reset!(integrator)
    integrator.u = integrator.p[2]
    global spike_count += 1
end

u0 = -75
tspan = (0.0, 40.0)
# p = (gL, EL, C, Vth, I, noise_std)
p = [10.0, -75.0, 5.0, -55.0, 0.0, 50.0]

threshold = DiscreteCallback(thr, reset!)
cb = CallbackSet(threshold)

I_values = range(0.0, stop=1000.0, length=300)
spiking_rates = []
Lif_sols = []

for I in I_values
    global spike_count = 0
    p[5] = I
    prob = ODEProblem(lif, u0, tspan, p, callback=cb)
    sol = solve(prob)
    push!(Lif_sols, sol)
    spiking_rate = spike_count / (tspan[2] - tspan[1])
    push!(spiking_rates, spiking_rate)
end

plot(I_values, spiking_rates, xlabel="Current (I)", ylabel="Spiking rate (Hz)", legend=false)



# Define the sigmoid function
sigmoid(x, p) = max.(p[1] ./ (1 .+ exp.(-p[2] * (x .- p[3]))), 0)

# Define the x / (x + 1) function
function x_over_x_plus_1(x, p)
    output = p[1] .* (x .+ p[3]) ./ ((x .+ p[3]) .+ p[2])
    condition = ((x .+ p[3]) .+ p[2]) .<= 0
    return max.(output .* (.!condition), 0)
end

# Perform the curve fit for sigmoid
p0_sigmoid = [1.0, 0.01, 100.0] # initial guess for sigmoid parameters
fit_sigmoid = curve_fit(sigmoid, I_values, spiking_rates, p0_sigmoid)

# Perform the curve fit for x / (x + 1)
p0_x_over_x_plus_1 = [1.0, 1.0, -100] # initial guess for x / (x + 1) parameters
fit_x_over_x_plus_1 = curve_fit(x_over_x_plus_1, I_values, spiking_rates, p0_x_over_x_plus_1)

# Composite function

interp_range = 8

I_values1 = I_values[I_values .<= 200]
spiking_rates1 = spiking_rates[1:length(I_values1)]
fit1 = curve_fit(sigmoid, I_values1, spiking_rates1, p0_sigmoid)

I_values2 = I_values[I_values .>= 200 + interp_range]
spiking_rates2 = spiking_rates[(length(I_values[I_values .<= 200 + interp_range])+1):end]
fit2 = curve_fit(x_over_x_plus_1, I_values2, spiking_rates2, p0_x_over_x_plus_1)

function composite_model(x, p1, p2)
    if x <= 200
        return sigmoid(x, p1)
    elseif x >= 200 + interp_range
        return x_over_x_plus_1(x, p2)
    else
        y1 = sigmoid(200, p1)
        y2 = x_over_x_plus_1(interp_range + 200, p2)
        slope = (y2 - y1) / interp_range
        return y1 + slope * (x - 200)
    end
end



# Plot the data and the fitted curves
I_plot = range(0.0, stop=1000.0, length=1000)
composite_I_values = [composite_model(x, fit1.param, fit2.param) for x in I_plot]

# Calculate the fitted values for the sigmoid and x_over_x_plus_1 functions
sigmoid_I_values = [sigmoid(x, fit_sigmoid.param) for x in I_plot]
x_over_x_plus_1_I_values = [x_over_x_plus_1(x, fit_x_over_x_plus_1.param) for x in I_plot]

# Plot the original data
fig = scatter(I_values, spiking_rates, label="Original data", color=:black, legend=:topleft)

# Plot the fitted sigmoid function
fig = plot!(I_plot, sigmoid_I_values, label="Sigmoid", linewidth=2, color=:blue)

# Plot the fitted x_over_x_plus_1 function
fig = plot!(I_plot, x_over_x_plus_1_I_values, label="x_over_x_plus_1", linewidth=2, color=:red)

# Plot the fitted composite function
fig = plot!(I_plot, composite_I_values, label="Composite sigmoid-xx1", linewidth=2, linestyle=:dash, color=:green)


fig = xlabel!("Current (I)")
fig = ylabel!("Spiking rate (Hz)")

fig

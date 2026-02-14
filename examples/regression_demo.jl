using Random
using LinearAlgebra
using LaTeXStrings
using PlotObjectiveFunction

Random.seed!(42)

n = 1000
p = 5
X = randn(n, p)

# Coefficients (intercept + 5 variables)
beta = [0.0, 1.0, 2.0, 3.0, 2.0, 1.0]
noise = 0.5 * randn(n)

y = beta[1] .+ X * beta[2:end] .+ noise
X_design = hcat(ones(n), X)

beta_hat = X_design \ y

objective(b) = sum((y .- X_design * b).^2)

param_names = [L"\beta_0", L"\beta_1", L"\beta_2", L"\beta_3", L"\beta_4", L"\beta_5"]

plot_objective(
    objective,
    beta_hat;
    param_names = param_names,
    outdir = "plots",
    basename = "objective_plot_reg5",
    output = :summary,
    resolution = 50,
)

using LaTeXStrings
using PlotObjectiveFunction

# Simple quadratic objective: minimum at x == beta
beta = [1.0, 2.0, 3.0, 2.0, 1.0]
objective(x) = sum((x .- beta) .^ 2)

param_names = [L"\beta_1", L"\beta_2", L"\beta_3", L"\beta_4", L"\beta_5"]

plot_objective(
    objective,
    beta;
    param_names = param_names,
    outdir = joinpath("plots", "quadratic5_demo"),
    basename = "objective_plot_quad5",
    output = :summary,
    resolution = 50,
)


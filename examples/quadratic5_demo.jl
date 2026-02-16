ENV["GKSwstype"] = "100"
ENV["GKS_WSTYPE"] = "100"

using PlotObjectiveFunction

function parse_optional_positive_int_env(variable_name::AbstractString)
    haskey(ENV, variable_name) || return nothing
    parsed_value = parse(Int, ENV[variable_name])
    parsed_value > 0 || throw(ArgumentError("$variable_name must be > 0"))
    return parsed_value
end

function parse_optional_bool_env(variable_name::AbstractString)
    haskey(ENV, variable_name) || return nothing
    normalized_value = lowercase(strip(ENV[variable_name]))
    if normalized_value in ("1", "true", "yes", "on")
        return true
    end
    if normalized_value in ("0", "false", "no", "off")
        return false
    end
    throw(ArgumentError("$variable_name must be true/false (or 1/0)"))
end

# Simple quadratic objective: minimum at x == beta
beta = [1.0, 2.0, 3.0, 2.0, 1.0]
objective(x) = sum((x .- beta) .^ 2)

param_names = ["beta1", "beta2", "beta3", "beta4", "beta5"]

result = run_plot_objective(
    objective,
    beta;
    param_names = param_names,
    outdir = get(ENV, "POF_OUTDIR", joinpath("plots", "quadratic5_demo")),
    basename = "objective_plot_quad5",
    output = :summary,
    resolution = parse(Int, get(ENV, "POF_RESOLUTION", "50")),
    backend = Symbol(get(ENV, "POF_BACKEND", "auto")),
    parallel = Symbol(get(ENV, "POF_PARALLEL", "auto")),
    threads_per_task = parse_optional_positive_int_env("POF_THREADS_PER_TASK"),
    blas_threads = parse_optional_positive_int_env("POF_BLAS_THREADS"),
    use_threads = parse_optional_bool_env("POF_USE_THREADS"),
)

println("summary = ", result.summary)
println("runtime = ", result.runtime)

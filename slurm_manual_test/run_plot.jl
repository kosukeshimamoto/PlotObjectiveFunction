ENV["GKSwstype"] = "100"
ENV["GKS_WSTYPE"] = "100"

using LinearAlgebra
using Random
using PlotObjectiveFunction

function parse_positive_int_env(variable_name::AbstractString, default_value::Integer)
    parsed_value = parse(Int, get(ENV, variable_name, string(default_value)))
    parsed_value > 0 || throw(ArgumentError("$variable_name must be > 0"))
    return parsed_value
end

function parse_positive_float_env(variable_name::AbstractString, default_value::Real)
    parsed_value = parse(Float64, get(ENV, variable_name, string(default_value)))
    parsed_value > 0 || throw(ArgumentError("$variable_name must be > 0"))
    return parsed_value
end

function parse_nonnegative_float_env(variable_name::AbstractString, default_value::Real)
    parsed_value = parse(Float64, get(ENV, variable_name, string(default_value)))
    parsed_value >= 0 || throw(ArgumentError("$variable_name must be >= 0"))
    return parsed_value
end

function parse_optional_positive_int_env(variable_name::AbstractString)
    haskey(ENV, variable_name) || return nothing
    parsed_value = parse(Int, ENV[variable_name])
    parsed_value > 0 || throw(ArgumentError("$variable_name must be > 0"))
    return parsed_value
end

function parse_optional_bool_env(variable_name::AbstractString)
    haskey(ENV, variable_name) || return nothing
    normalized_value = lowercase(strip(ENV[variable_name]))
    normalized_value in ("1", "true", "yes", "on") && return true
    normalized_value in ("0", "false", "no", "off") && return false
    throw(ArgumentError("$variable_name must be true/false (or 1/0)"))
end

function parse_symbol_env(
    variable_name::AbstractString,
    default_value::Symbol,
    allowed_values::Tuple{Vararg{Symbol}},
)
    parsed_value = Symbol(get(ENV, variable_name, String(default_value)))
    parsed_value in allowed_values ||
        throw(ArgumentError("$variable_name must be one of $(collect(allowed_values))"))
    return parsed_value
end

function build_regression_objective(
    num_obs::Integer,
    num_features::Integer;
    seed::Integer,
    noise_scale::Real,
)
    Random.seed!(seed)

    explanatory_vars = randn(num_obs, num_features)
    true_parameter_values = [0.5; collect(range(0.8, step = 0.3, length = num_features))]
    response_noise = noise_scale * randn(num_obs)
    observed_values = true_parameter_values[1] .+ explanatory_vars * true_parameter_values[2:end] .+ response_noise
    design_matrix = hcat(ones(num_obs), explanatory_vars)

    objective_function(parameter_values) = sum((observed_values .- design_matrix * parameter_values) .^ 2)
    estimated_parameter_values = design_matrix \ observed_values
    parameter_names = ["beta$(parameter_index - 1)" for parameter_index in eachindex(estimated_parameter_values)]

    return objective_function, estimated_parameter_values, parameter_names
end

function run_manual_slurm_plot()
    num_obs = parse_positive_int_env("POF_NUM_OBS", 3000)
    num_features = parse_positive_int_env("POF_NUM_FEATURES", 8)
    seed = parse_positive_int_env("POF_SEED", 42)
    resolution = parse_positive_int_env("POF_RESOLUTION", 50)
    noise_scale = parse_positive_float_env("POF_NOISE_SCALE", 0.5)
    range_frac = parse_nonnegative_float_env("POF_RANGE_FRAC", 0.5)

    objective_function, estimated_parameter_values, parameter_names =
        build_regression_objective(
            num_obs,
            num_features;
            seed = seed,
            noise_scale = noise_scale,
        )

    result = run_plot_objective(
        objective_function,
        estimated_parameter_values;
        param_names = parameter_names,
        outdir = get(ENV, "POF_OUTDIR", joinpath("slurm_manual_test", "outputs")),
        basename = get(ENV, "POF_BASENAME", "objective_plot_manual"),
        summary_name = get(ENV, "POF_SUMMARY_NAME", "objective_plot_manual_summary"),
        output = :summary,
        resolution = resolution,
        range_frac = range_frac,
        backend = parse_symbol_env("POF_BACKEND", :auto, (:auto, :local, :slurm)),
        parallel = parse_symbol_env("POF_PARALLEL", :auto, (:manual, :auto)),
        threads_per_task = parse_optional_positive_int_env("POF_THREADS_PER_TASK"),
        blas_threads = parse_optional_positive_int_env("POF_BLAS_THREADS"),
        use_threads = parse_optional_bool_env("POF_USE_THREADS"),
        auto_samples_per_thread = parse_positive_int_env("POF_AUTO_SAMPLES_PER_THREAD", 2),
        auto_repeats = parse_positive_int_env("POF_AUTO_REPEATS", 2),
        auto_min_speedup = parse_positive_float_env("POF_AUTO_MIN_SPEEDUP", 1.05),
    )

    println("summary_file = ", result.summary)
    println("runtime_config = ", result.runtime)
end

run_manual_slurm_plot()

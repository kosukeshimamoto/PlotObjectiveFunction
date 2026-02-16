module PlotObjectiveFunction

export plot_objective,
    run_plot_objective,
    describe_output_combinations,
    print_output_combinations,
    write_output_combinations

using LinearAlgebra
using Plots
import GR

"""
    plot_objective(f, x0; range_frac=0.5, resolution=50, outdir="plots",
                   basename="objective_plot", summary_name=nothing,
                   output=:summary, one_d=true, two_d=true, zero_range=1.0,
                   param_names=nothing,
                   split_summary_pages=false,
                   colormap=cgrad(:viridis, 256),
                   contour_levels=10,
                   xtick_count=3,
                   show_reference_legend=true,
                   use_threads=false,
                   linecolor=:royalblue, refcolor=:darkorange)

Plot an objective function around `x0`.

Arguments:
- `f`: objective function that accepts an `AbstractVector` and returns a real number.
- `x0`: initial/true parameter vector.

Keyword arguments:
- `range_frac`: range as ±fraction around each parameter value.
- `resolution`: number of grid points per axis (default: 50). 1D uses `n` points, 2D uses `n^2` evaluations.
- `outdir`: output directory for PDFs (default: "plots").
- `basename`: base filename prefix (default: "objective_plot").
- `output`: `:summary`, `:individual`, or `:both` (default: `:summary`).
- `summary_name`: override the summary PDF filename (optional).
- `param_names`: parameter names for titles (length must match `x0`).
- `one_d`: generate 1D plots for each parameter.
- `two_d`: generate 2D heatmaps for parameter combinations (i < j).
- `zero_range`: absolute range used when a parameter value is zero.
- `split_summary_pages`: if true and plots exceed one page (15 plots), write one PDF per page
  instead of a single multi-page PDF.
- `colormap`: color map for heatmaps (default: viridis).
- `contour_levels`: number of contour levels to overlay on 2D heatmaps (default: 10). Set to 0 to disable.
- `xtick_count`: number of x-axis tick labels to show (default: 3).
- `show_reference_legend`: show `True`/`Min` legend entries (default: true).
- `use_threads`: evaluate objective values with `Threads.@threads` (default: false).
  When enabled, `f` must be thread-safe.
- `linecolor`: line color for 1D plots (default: blue).
- `refcolor`: reference line color for `x0` (default: orange).

The summary PDF uses A4-sized pages with 3 columns and 4-5 rows.
Plots are ordered as 1D first, then 2D, without forcing a page break.

Returns a named tuple with generated file paths:
`(summary=..., one_d=..., two_d=...)`.

If `split_summary_pages=true` and the summary exceeds one page, `summary` is a vector of page PDF paths.
"""
function plot_objective(
    f,
    x0::AbstractVector;
    range_frac::Real = 0.5,
    resolution::Integer = 50,
    outdir::AbstractString = "plots",
    basename::AbstractString = "objective_plot",
    summary_name::Union{Nothing, AbstractString} = nothing,
    output::Symbol = :summary,
    one_d::Bool = true,
    two_d::Bool = true,
    zero_range::Real = 1.0,
    param_names::Union{Nothing, AbstractVector{<:AbstractString}} = nothing,
    split_summary_pages::Bool = false,
    colormap = cgrad(:viridis, 256),
    contour_levels::Integer = 10,
    xtick_count::Integer = 3,
    show_reference_legend::Bool = true,
    use_threads::Bool = false,
    linecolor = :royalblue,
    refcolor = :darkorange,
)
    resolution >= 2 || throw(ArgumentError("resolution must be >= 2"))
    range_frac >= 0 || throw(ArgumentError("range_frac must be >= 0"))
    zero_range > 0 || throw(ArgumentError("zero_range must be > 0"))
    output in (:summary, :individual, :both) ||
        throw(ArgumentError("output must be :summary, :individual, or :both"))

    mkpath(outdir)
    gr()

    x0v = collect(x0)
    n = length(x0v)
    if param_names !== nothing && length(param_names) != n
        throw(ArgumentError("param_names length must match x0 length"))
    end
    names = param_names === nothing ? ["Param$(i)" for i in 1:n] : collect(param_names)
    xtick_count >= 2 || throw(ArgumentError("xtick_count must be >= 2"))
    contour_levels >= 0 || throw(ArgumentError("contour_levels must be >= 0"))
    legend_position = show_reference_legend ? :topright : false
    true_label = show_reference_legend ? "True" : ""
    min_label = show_reference_legend ? "Min" : ""

    function param_range(x)
        if iszero(x)
            lo = -zero_range
            hi = zero_range
        else
            lo = x * (1 - range_frac)
            hi = x * (1 + range_frac)
        end

        if lo == hi
            delta = max(abs(x) * range_frac, zero_range)
            lo = x - delta
            hi = x + delta
        end

        if lo > hi
            lo, hi = hi, lo
        end

        # Ensure `x` is included in the grid even when `resolution` is even,
        # so the reference point and the sampled minimum can coincide.
        if resolution <= 2
            return collect(range(lo, hi, length = resolution))
        end

        n_left = (resolution - 1) ÷ 2
        n_right = resolution - n_left - 1

        left = range(lo, x, length = n_left + 1)
        right = range(x, hi, length = n_right + 1)

        xs = Vector{Float64}(undef, resolution)
        k = 1
        for idx in 1:n_left
            xs[k] = left[idx]
            k += 1
        end
        for idx in 1:(n_right + 1)
            xs[k] = right[idx]
            k += 1
        end
        return xs
    end

    one_d_plots = Plots.Plot[]
    two_d_plots = Plots.Plot[]
    one_d_files = String[]
    two_d_files = String[]
    base_plot_style = (
        guidefontsize = 10,
        tickfontsize = 10,
        titlefontsize = 10,
        legendfontsize = 7,
        left_margin = 3 * Plots.mm,
        right_margin = 3 * Plots.mm,
        top_margin = 3 * Plots.mm,
        bottom_margin = 3 * Plots.mm,
    )

    if one_d
        for i in 1:n
            xs = param_range(x0v[i])
            yvals = Vector{Float64}(undef, length(xs))
            if use_threads
                Threads.@threads for sample_index in eachindex(xs)
                    parameter_values = copy(x0v)
                    parameter_values[i] = xs[sample_index]
                    yvals[sample_index] = Float64(f(parameter_values))
                end
            else
                parameter_values = copy(x0v)
                for (sample_index, sample_value) in enumerate(xs)
                    parameter_values[i] = sample_value
                    yvals[sample_index] = Float64(f(parameter_values))
                end
            end

            plt = plot(
                xs,
                yvals;
                title = names[i],
                legend = legend_position,
                label = "",
                color = linecolor,
                linewidth = 0.6,
                xticks = range(first(xs), last(xs), length = xtick_count),
                base_plot_style...,
            )
            scatter!(
                plt,
                xs,
                yvals;
                color = linecolor,
                markersize = 3,
                markerstrokewidth = 0,
                label = "",
            )
            vline!(plt, [x0v[i]], color = refcolor, linewidth = 1.5, label = true_label)
            ymin, idx = findmin(yvals)
            scatter!(
                plt,
                [xs[idx]],
                [ymin];
                markersize = 6,
                marker = :diamond,
                color = linecolor,
                markerstrokewidth = 0,
                label = min_label,
            )
            push!(one_d_plots, plt)

            if output in (:individual, :both)
                path = joinpath(outdir, "$(basename)_1d_p$(i).pdf")
                savefig(plt, path)
                push!(one_d_files, path)
            end
        end
    end

    if two_d && n >= 2
        want_summary = output in (:summary, :both)
        want_individual = output in (:individual, :both)

        for i in 1:(n - 1)
            for j in (i + 1):n
                xs = param_range(x0v[i])
                ys = param_range(x0v[j])
                z = Matrix{Float64}(undef, length(ys), length(xs))
                cartesian_grid_indices = CartesianIndices(z)
                if use_threads
                    Threads.@threads for linear_index in eachindex(cartesian_grid_indices)
                        grid_index = cartesian_grid_indices[linear_index]
                        row_index = grid_index[1]
                        col_index = grid_index[2]
                        parameter_values = copy(x0v)
                        parameter_values[i] = xs[col_index]
                        parameter_values[j] = ys[row_index]
                        z[row_index, col_index] = Float64(f(parameter_values))
                    end
                else
                    parameter_values = copy(x0v)
                    for grid_index in cartesian_grid_indices
                        row_index = grid_index[1]
                        col_index = grid_index[2]
                        parameter_values[i] = xs[col_index]
                        parameter_values[j] = ys[row_index]
                        z[row_index, col_index] = Float64(f(parameter_values))
                    end
                end

                dx_lo = xs[2] - xs[1]
                dx_hi = xs[end] - xs[end - 1]
                dy_lo = ys[2] - ys[1]
                dy_hi = ys[end] - ys[end - 1]
                zmin, idx = findmin(z)
                iy, ix = Tuple(CartesianIndices(z)[idx])

                function build_2d_plot(; show_colorbar::Bool)
                    plt = heatmap(
                        xs,
                        ys,
                        z;
                        title = "($(names[i]), $(names[j]))",
                        color = colormap,
                        colorbar = show_colorbar,
                        legend = legend_position,
                        label = "",
                        # Smooth the background so contours (interpolated) don't look offset vs. the heatmap.
                        interpolate = true,
                        # Make 2D plots visually consistent (do not enforce unit aspect ratio).
                        aspect_ratio = :none,
                        widen = false,
                        xlims = (first(xs) - dx_lo / 2, last(xs) + dx_hi / 2),
                        ylims = (first(ys) - dy_lo / 2, last(ys) + dy_hi / 2),
                        xticks = range(first(xs), last(xs), length = xtick_count),
                        base_plot_style...,
                    )
                    if contour_levels > 0
                        contour!(
                            plt,
                            xs,
                            ys,
                            z;
                            levels = contour_levels,
                            linecolor = :black,
                            linewidth = 0.15,
                            linealpha = 0.12,
                            label = "",
                        )
                    end
                    vline!(plt, [x0v[i]], color = refcolor, linewidth = 1.0, label = true_label)
                    hline!(plt, [x0v[j]], color = refcolor, linewidth = 1.0, label = "")
                    scatter!(
                        plt,
                        [xs[ix]],
                        [ys[iy]];
                        markersize = 5,
                        marker = :diamond,
                        color = linecolor,
                        markerstrokewidth = 0,
                        label = min_label,
                    )
                    return plt
                end

                if want_summary
                    push!(two_d_plots, build_2d_plot(; show_colorbar = false))
                end

                if want_individual
                    plt = build_2d_plot(; show_colorbar = true)
                    path = joinpath(outdir, "$(basename)_2d_p$(i)_p$(j).pdf")
                    savefig(plt, path)
                    push!(two_d_files, path)
                end
            end
        end
    end

    summary_file = nothing
    if output in (:summary, :both)
        summary_path = if summary_name === nothing
            joinpath(outdir, "$(basename).pdf")
        else
            joinpath(outdir, endswith(summary_name, ".pdf") ? summary_name : summary_name * ".pdf")
        end

        mkpath(dirname(summary_path))
        all_plots = vcat(one_d_plots, two_d_plots)
        isempty(all_plots) && throw(ArgumentError("no plots to write: set one_d=true and/or two_d=true"))

        pages = _build_pages(all_plots)
        if split_summary_pages && length(pages) > 1
            summary_root, _ = splitext(summary_path)
            summary_files = String[]
            sizehint!(summary_files, length(pages))
            for (k, p) in enumerate(pages)
                page_path = summary_root * "_page$(k).pdf"
                savefig(p, page_path)
                push!(summary_files, page_path)
            end
            summary_file = summary_files
        elseif split_summary_pages
            savefig(only(pages), summary_path)
            summary_file = summary_path
        else
            _write_summary_pdf(pages, summary_path)
            summary_file = summary_path
        end
    end

    return (summary = summary_file, one_d = one_d_files, two_d = two_d_files)
end

"""
    run_plot_objective(f, x0; backend=:auto, parallel=:manual, threads_per_task=nothing,
                       blas_threads=nothing, use_threads=nothing,
                       auto_samples_per_thread=2, auto_repeats=2, auto_min_speedup=1.05,
                       kwargs...)

Run `plot_objective` with runtime settings shared by local and SLURM execution.

Keyword arguments:
- `backend`: `:auto`, `:local`, or `:slurm`.
- `parallel`: `:manual` or `:auto`.
- `threads_per_task`: CPU budget for one Julia process.
- `blas_threads`: BLAS thread count inside each objective evaluation.
- `use_threads`: enable/disable `Threads.@threads` for objective grids.
- `auto_samples_per_thread`: benchmark sample count per Julia thread when `parallel=:auto`.
- `auto_repeats`: benchmark repetitions when `parallel=:auto`.
- `auto_min_speedup`: minimum speedup required to enable threaded grid evaluation.

Additional keyword arguments are forwarded to `plot_objective`.

Returns a named tuple:
`(summary=..., one_d=..., two_d=..., runtime=...)`.
"""
function run_plot_objective(
    f,
    x0::AbstractVector;
    backend::Symbol = :auto,
    parallel::Symbol = :manual,
    threads_per_task::Union{Nothing, Integer} = nothing,
    blas_threads::Union{Nothing, Integer} = nothing,
    use_threads::Union{Nothing, Bool} = nothing,
    auto_samples_per_thread::Integer = 2,
    auto_repeats::Integer = 2,
    auto_min_speedup::Real = 1.05,
    kwargs...,
)
    resolved_backend = _resolve_backend(backend)
    resolved_threads_per_task = _resolve_threads_per_task(resolved_backend, threads_per_task)
    resolved_use_threads, resolved_blas_threads, auto_detail = _resolve_parallel_configuration(
        f,
        x0;
        parallel = parallel,
        threads_per_task = resolved_threads_per_task,
        blas_threads = blas_threads,
        use_threads = use_threads,
        auto_samples_per_thread = auto_samples_per_thread,
        auto_repeats = auto_repeats,
        auto_min_speedup = auto_min_speedup,
    )

    previous_blas_threads = LinearAlgebra.BLAS.get_num_threads()
    resolved_blas_threads != previous_blas_threads && LinearAlgebra.BLAS.set_num_threads(resolved_blas_threads)
    try
        plot_result = plot_objective(f, x0; use_threads = resolved_use_threads, kwargs...)
        return (
            summary = plot_result.summary,
            one_d = plot_result.one_d,
            two_d = plot_result.two_d,
            runtime = (
                backend = resolved_backend,
                parallel = parallel,
                threads_per_task = resolved_threads_per_task,
                julia_threads = Threads.nthreads(),
                blas_threads = resolved_blas_threads,
                use_threads = resolved_use_threads,
                auto = auto_detail,
            ),
        )
    finally
        resolved_blas_threads != previous_blas_threads &&
            LinearAlgebra.BLAS.set_num_threads(previous_blas_threads)
    end
end

function _resolve_backend(backend::Symbol)
    backend == :auto && return haskey(ENV, "SLURM_JOB_ID") ? :slurm : :local
    backend in (:local, :slurm) || throw(ArgumentError("backend must be :auto, :local, or :slurm"))
    return backend
end

function _parse_positive_int(value::AbstractString, variable_name::AbstractString)
    parsed_value = try
        parse(Int, value)
    catch
        throw(ArgumentError("$variable_name must be a positive integer, got '$value'"))
    end
    parsed_value > 0 || throw(ArgumentError("$variable_name must be a positive integer, got '$value'"))
    return parsed_value
end

function _resolve_threads_per_task(backend::Symbol, threads_per_task::Union{Nothing, Integer})
    if threads_per_task !== nothing
        threads_per_task > 0 || throw(ArgumentError("threads_per_task must be > 0"))
        return threads_per_task
    end
    if backend == :slurm && haskey(ENV, "SLURM_CPUS_PER_TASK")
        return _parse_positive_int(ENV["SLURM_CPUS_PER_TASK"], "SLURM_CPUS_PER_TASK")
    end
    return Threads.nthreads()
end

function _resolve_parallel_configuration(
    f,
    x0::AbstractVector;
    parallel::Symbol,
    threads_per_task::Integer,
    blas_threads::Union{Nothing, Integer},
    use_threads::Union{Nothing, Bool},
    auto_samples_per_thread::Integer,
    auto_repeats::Integer,
    auto_min_speedup::Real,
)
    parallel in (:manual, :auto) || throw(ArgumentError("parallel must be :manual or :auto"))
    if parallel == :manual
        return _resolve_manual_parallel_configuration(
            ;
            threads_per_task = threads_per_task,
            blas_threads = blas_threads,
            use_threads = use_threads,
        )
    end
    return _resolve_auto_parallel_configuration(
        f,
        x0;
        threads_per_task = threads_per_task,
        blas_threads = blas_threads,
        use_threads = use_threads,
        auto_samples_per_thread = auto_samples_per_thread,
        auto_repeats = auto_repeats,
        auto_min_speedup = auto_min_speedup,
    )
end

function _resolve_manual_parallel_configuration(;
    threads_per_task::Integer,
    blas_threads::Union{Nothing, Integer},
    use_threads::Union{Nothing, Bool},
)
    resolved_blas_threads = something(blas_threads, 1)
    resolved_blas_threads > 0 || throw(ArgumentError("blas_threads must be > 0"))
    resolved_blas_threads <= threads_per_task ||
        throw(ArgumentError("blas_threads must be <= threads_per_task"))

    resolved_use_threads = something(use_threads, false)
    resolved_use_threads && Threads.nthreads() <= 1 &&
        throw(ArgumentError("use_threads=true requires JULIA_NUM_THREADS > 1"))

    return resolved_use_threads, resolved_blas_threads, (strategy = :manual,)
end

function _resolve_auto_parallel_configuration(
    f,
    x0::AbstractVector;
    threads_per_task::Integer,
    blas_threads::Union{Nothing, Integer},
    use_threads::Union{Nothing, Bool},
    auto_samples_per_thread::Integer,
    auto_repeats::Integer,
    auto_min_speedup::Real,
)
    auto_samples_per_thread > 0 || throw(ArgumentError("auto_samples_per_thread must be > 0"))
    auto_repeats > 0 || throw(ArgumentError("auto_repeats must be > 0"))
    auto_min_speedup >= 1 || throw(ArgumentError("auto_min_speedup must be >= 1"))

    selected_blas_threads = if blas_threads === nothing
        _select_auto_blas_threads(
            f,
            x0;
            threads_per_task = threads_per_task,
            auto_repeats = auto_repeats,
        )
    else
        blas_threads
    end
    selected_blas_threads > 0 || throw(ArgumentError("blas_threads must be > 0"))
    selected_blas_threads <= threads_per_task ||
        throw(ArgumentError("blas_threads must be <= threads_per_task"))

    benchmark_parameter_values = _build_benchmark_parameter_values(x0, auto_samples_per_thread)
    previous_blas_threads = LinearAlgebra.BLAS.get_num_threads()
    selected_use_threads = false
    serial_seconds = Inf
    threaded_seconds = Inf
    try
        LinearAlgebra.BLAS.set_num_threads(selected_blas_threads)
        serial_seconds = _benchmark_serial_grid(f, benchmark_parameter_values, auto_repeats)
        threaded_seconds = Threads.nthreads() > 1 ?
            _benchmark_threaded_grid(f, benchmark_parameter_values, auto_repeats) : Inf
        selected_use_threads = if use_threads === nothing
            Threads.nthreads() > 1 &&
                isfinite(threaded_seconds) &&
                serial_seconds / threaded_seconds >= auto_min_speedup
        else
            use_threads
        end
    finally
        LinearAlgebra.BLAS.set_num_threads(previous_blas_threads)
    end
    selected_use_threads && Threads.nthreads() <= 1 &&
        throw(ArgumentError("use_threads=true requires JULIA_NUM_THREADS > 1"))

    return selected_use_threads, selected_blas_threads, (
        strategy = :auto,
        serial_seconds = serial_seconds,
        threaded_seconds = isfinite(threaded_seconds) ? threaded_seconds : nothing,
        selected = selected_use_threads ? :threaded_grid : :serial_grid,
    )
end

function _select_auto_blas_threads(
    f,
    x0::AbstractVector;
    threads_per_task::Integer,
    auto_repeats::Integer,
)
    threads_per_task == 1 && return 1
    benchmark_input = collect(x0)
    max_blas_threads = threads_per_task
    candidate_threads = max_blas_threads == 1 ? (1,) : (1, max_blas_threads)
    previous_blas_threads = LinearAlgebra.BLAS.get_num_threads()
    selected_blas_threads = first(candidate_threads)
    best_seconds = Inf
    try
        for candidate in candidate_threads
            LinearAlgebra.BLAS.set_num_threads(candidate)
            candidate_seconds = _benchmark_single_eval(f, benchmark_input, auto_repeats)
            if candidate_seconds < best_seconds
                best_seconds = candidate_seconds
                selected_blas_threads = candidate
            end
        end
    finally
        LinearAlgebra.BLAS.set_num_threads(previous_blas_threads)
    end
    return selected_blas_threads
end

function _build_benchmark_parameter_values(x0::AbstractVector, auto_samples_per_thread::Integer)
    if isempty(x0)
        return [Float64[]]
    end

    num_samples = max(4, auto_samples_per_thread * max(2, Threads.nthreads()))
    benchmark_parameter_values = Vector{Vector{Float64}}(undef, num_samples)
    center_value = Float64(x0[1])
    delta = iszero(center_value) ? 1.0 : abs(center_value) / 10
    perturbations = range(-delta, delta, length = num_samples)
    for sample_index in eachindex(perturbations)
        parameter_values = Float64.(x0)
        parameter_values[1] = center_value + perturbations[sample_index]
        benchmark_parameter_values[sample_index] = parameter_values
    end
    return benchmark_parameter_values
end

function _benchmark_serial_grid(
    f,
    benchmark_parameter_values::Vector{Vector{Float64}},
    auto_repeats::Integer,
)
    best_seconds = Inf
    for _ in 1:auto_repeats
        start_ns = time_ns()
        total_objective = 0.0
        for parameter_values in benchmark_parameter_values
            total_objective += Float64(f(parameter_values))
        end
        elapsed_seconds = (time_ns() - start_ns) / 1e9
        isfinite(total_objective) || throw(ArgumentError("objective function returned non-finite value"))
        best_seconds = min(best_seconds, elapsed_seconds)
    end
    return best_seconds
end

function _benchmark_threaded_grid(
    f,
    benchmark_parameter_values::Vector{Vector{Float64}},
    auto_repeats::Integer,
)
    Threads.nthreads() <= 1 && return Inf
    best_seconds = Inf
    for _ in 1:auto_repeats
        partial_objectives = zeros(Float64, Threads.nthreads())
        start_ns = time_ns()
        Threads.@threads for sample_index in eachindex(benchmark_parameter_values)
            partial_objectives[Threads.threadid()] += Float64(f(benchmark_parameter_values[sample_index]))
        end
        elapsed_seconds = (time_ns() - start_ns) / 1e9
        isfinite(sum(partial_objectives)) || throw(ArgumentError("objective function returned non-finite value"))
        best_seconds = min(best_seconds, elapsed_seconds)
    end
    return best_seconds
end

function _benchmark_single_eval(f, x0::AbstractVector, auto_repeats::Integer)
    best_seconds = Inf
    for _ in 1:auto_repeats
        start_ns = time_ns()
        objective_value = Float64(f(copy(x0)))
        elapsed_seconds = (time_ns() - start_ns) / 1e9
        isfinite(objective_value) || throw(ArgumentError("objective function returned non-finite value"))
        best_seconds = min(best_seconds, elapsed_seconds)
    end
    return best_seconds
end

"""
    describe_output_combinations(x0; outdir="plots", basename="objective_plot", summary_name=nothing)

Dry-run helper that describes what files would be written for all combinations of:
`output ∈ (:summary, :individual, :both)`, `one_d ∈ (false, true)`, `two_d ∈ (false, true)`,
`split_summary_pages ∈ (false, true)`.

Meaningless combinations that would produce no plots are skipped.
Also, `split_summary_pages` is only enumerated for `output != :individual`.

Returns a vector of named tuples. No files are created.
"""
function describe_output_combinations(
    x0::AbstractVector;
    outdir::AbstractString = "plots",
    basename::AbstractString = "objective_plot",
    summary_name::Union{Nothing, AbstractString} = nothing,
)
    n = length(x0)
    outputs = (:summary, :individual, :both)
    bools = (false, true)

    summary_path = if summary_name === nothing
        joinpath(outdir, "$(basename).pdf")
    else
        joinpath(outdir, endswith(summary_name, ".pdf") ? summary_name : summary_name * ".pdf")
    end
    summary_root, _ = splitext(summary_path)

    combos = NamedTuple[]
    sizehint!(combos, length(outputs) * length(bools)^3)

    for output in outputs, one_d in bools, two_d in bools
        nplots_1d = one_d ? n : 0
        nplots_2d = (two_d && n >= 2) ? (n * (n - 1)) ÷ 2 : 0
        nplots = nplots_1d + nplots_2d
        nplots == 0 && continue
        npages = nplots == 0 ? 0 : cld(nplots, 15)

        split_bools = output == :individual ? (false,) : bools
        for split_summary_pages in split_bools
            wants_summary = output in (:summary, :both)
            will_error = false

            summary_files = String[]
            if wants_summary
                if split_summary_pages && npages > 1
                    sizehint!(summary_files, npages)
                    for k in 1:npages
                        push!(summary_files, summary_root * "_page$(k).pdf")
                    end
                else
                    push!(summary_files, summary_path)
                end
            end

            wants_individual = output in (:individual, :both)
            one_d_count = wants_individual ? nplots_1d : 0
            two_d_count = wants_individual ? nplots_2d : 0

            push!(
                combos,
                (
                    output = output,
                    one_d = one_d,
                    two_d = two_d,
                    split_summary_pages = split_summary_pages,
                    nparams = n,
                    nplots_1d = nplots_1d,
                    nplots_2d = nplots_2d,
                    nplots = nplots,
                    npages = npages,
                    summary_files = summary_files,
                    summary_is_multipage_pdf = wants_summary && !split_summary_pages && npages > 1,
                    one_d_count = one_d_count,
                    two_d_count = two_d_count,
                    will_error = will_error,
                ),
            )
        end
    end

    return combos
end

"""
    print_output_combinations(x0; kwargs...)

Pretty-prints `describe_output_combinations(x0; kwargs...)` to stdout.
"""
function print_output_combinations(x0::AbstractVector; kwargs...)
    combos = describe_output_combinations(x0; kwargs...)
    for c in combos
        println(
            "output=",
            c.output,
            " one_d=",
            c.one_d,
            " two_d=",
            c.two_d,
            " split_summary_pages=",
            c.split_summary_pages,
            " | nplots=",
            c.nplots,
            " npages=",
            c.npages,
            " | summary_files=",
            isempty(c.summary_files) ? "[]" : c.summary_files,
            " | individual=(1d:",
            c.one_d_count,
            ", 2d:",
            c.two_d_count,
            ")",
            c.will_error ? " | ERROR(no plots for summary)" : "",
        )
    end
    return nothing
end

"""
    write_output_combinations(f, x0; outdir="plots/output_combinations", basename="objective_plot", summary_name=nothing, kwargs...)

Write plots for all combinations of output-related options into subfolders under `outdir`.
This is mainly for inspecting layouts and outputs; it can create many files.

Combinations:
- `output ∈ (:summary, :individual, :both)`
- `one_d ∈ (false, true)`
- `two_d ∈ (false, true)`
- `split_summary_pages ∈ (false, true)`

Meaningless combinations that would produce no plots are skipped.
Also, `split_summary_pages` is only enumerated for `output != :individual`.

Folders are organized as:
`outdir/output_<output>/1d_<on|off>_2d_<on|off>/split_<on|off>/` (summary/both)
`outdir/output_individual/1d_<on|off>_2d_<on|off>/` (individual only)

Returns a vector of named tuples describing each attempted combination and its result.
"""
function write_output_combinations(
    f,
    x0::AbstractVector;
    outdir::AbstractString = joinpath("plots", "output_combinations"),
    basename::AbstractString = "objective_plot",
    summary_name::Union{Nothing, AbstractString} = nothing,
    kwargs...,
)
    outputs = (:summary, :individual, :both)
    bools = (false, true)
    results = NamedTuple[]
    sizehint!(results, length(outputs) * length(bools)^3)

    n = length(x0)
    for output in outputs, one_d in bools, two_d in bools
        nplots_1d = one_d ? n : 0
        nplots_2d = (two_d && n >= 2) ? (n * (n - 1)) ÷ 2 : 0
        nplots = nplots_1d + nplots_2d
        nplots == 0 && continue

        split_bools = output == :individual ? (false,) : bools
        for split_summary_pages in split_bools
            combo_dir = if output == :individual
                joinpath(
                    outdir,
                    "output_$(output)",
                    "1d_$(one_d ? "on" : "off")_2d_$(two_d ? "on" : "off")",
                )
            else
                joinpath(
                    outdir,
                    "output_$(output)",
                    "1d_$(one_d ? "on" : "off")_2d_$(two_d ? "on" : "off")",
                    "split_$(split_summary_pages ? "on" : "off")",
                )
            end
            mkpath(combo_dir)

            res = plot_objective(
                f,
                x0;
                outdir = combo_dir,
                basename = basename,
                summary_name = summary_name,
                output = output,
                one_d = one_d,
                two_d = two_d,
                split_summary_pages = split_summary_pages,
                kwargs...,
            )

            push!(
                results,
                (
                    output = output,
                    one_d = one_d,
                    two_d = two_d,
                    split_summary_pages = split_summary_pages,
                    outdir = combo_dir,
                    result = res,
                ),
            )
        end
    end

    return results
end

function _build_pages(plots::Vector{Plots.Plot};
    min_rows::Int = 4, max_rows::Int = 5, cols::Int = 3, pad_rows::Int = 1, pad_cols::Int = 1,
    pad_ratio::Real = 0.2)
    pad_ratio > 0 || throw(ArgumentError("pad_ratio must be > 0"))
    pages = Plots.Plot[]
    page_size = max_rows * cols
    total = length(plots)
    fixed_rows = total > page_size ? max_rows : min(max_rows, max(min_rows, cld(total, cols)))
    idx = 1

    while idx <= total
        chunk = plots[idx:min(idx + page_size - 1, total)]
        rows = fixed_rows
        total_rows = rows + 2 * pad_rows
        total_cols = cols + 2 * pad_cols
        slots = Plots.Plot[]
        sizehint!(slots, total_rows * total_cols)
        plot_idx = 1

        for r in 1:total_rows
            for c in 1:total_cols
                if r <= pad_rows || r > pad_rows + rows || c <= pad_cols || c > pad_cols + cols
                    push!(slots, plot(; framestyle = :none, ticks = false, grid = false, axis = false))
                elseif plot_idx <= length(chunk)
                    push!(slots, chunk[plot_idx])
                    plot_idx += 1
                else
                    push!(slots, plot(; framestyle = :none, ticks = false, grid = false, axis = false))
                end
            end
        end

        heights = if pad_rows > 0
            weights = vcat(fill(pad_ratio, pad_rows), fill(1.0, rows), fill(pad_ratio, pad_rows))
            weights ./ sum(weights)
        else
            nothing
        end
        widths = if pad_cols > 0
            weights = vcat(fill(pad_ratio, pad_cols), fill(1.0, cols), fill(pad_ratio, pad_cols))
            weights ./ sum(weights)
        else
            nothing
        end
        layout = if heights === nothing && widths === nothing
            grid(total_rows, total_cols)
        elseif heights === nothing
            grid(total_rows, total_cols; widths = widths)
        elseif widths === nothing
            grid(total_rows, total_cols; heights = heights)
        else
            grid(total_rows, total_cols; heights = heights, widths = widths)
        end

        page = plot(
            slots...,
            layout = layout,
            size = (1200, 1697),
            margin = 0 * Plots.mm,
        )
        push!(pages, page)
        idx += page_size
    end

    return pages
end

function _write_summary_pdf(pages::Vector{Plots.Plot}, path::AbstractString)
    GR.beginprint(path)
    try
        for p in pages
            display(p)
            GR.updatews()
        end
    finally
        GR.endprint()
    end
end

end

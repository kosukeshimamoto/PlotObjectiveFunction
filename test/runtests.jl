ENV["GKSwstype"] = "100"

using Test
using PlotObjectiveFunction

@testset "plot_objective" begin
    f(x) = sum((x .- 1.0).^2)
    x0 = [1.0, 1.0, 1.0]
    outdir = mktempdir()

    names = ["Param1", "Param2", "Param3"]
    result = plot_objective(
        f,
        x0;
        outdir = outdir,
        basename = "test",
        resolution = 10,
        range_frac = 0.5,
        output = :both,
        param_names = names,
        summary_name = "summary_custom",
    )

    @test result.summary == joinpath(outdir, "summary_custom.pdf")
    @test isfile(result.summary)
    @test length(result.one_d) == length(x0)
    @test length(result.two_d) == length(x0) * (length(x0) - 1) ÷ 2
    @test all(isfile, result.one_d)
    @test all(isfile, result.two_d)

    hidden_legend_result = plot_objective(
        f,
        x0;
        outdir = outdir,
        basename = "test_no_legend",
        resolution = 8,
        output = :both,
        show_reference_legend = false,
    )
    @test isfile(hidden_legend_result.summary)
    @test length(hidden_legend_result.one_d) == length(x0)
    @test length(hidden_legend_result.two_d) == length(x0) * (length(x0) - 1) ÷ 2

    threaded_result = plot_objective(
        f,
        x0;
        outdir = outdir,
        basename = "test_threads",
        resolution = 8,
        output = :both,
        use_threads = true,
    )
    @test isfile(threaded_result.summary)
    @test length(threaded_result.one_d) == length(x0)
    @test length(threaded_result.two_d) == length(x0) * (length(x0) - 1) ÷ 2
    @test all(isfile, threaded_result.one_d)
    @test all(isfile, threaded_result.two_d)

    @test_throws ArgumentError plot_objective(
        f,
        x0;
        outdir = outdir,
        resolution = 5,
        param_names = ["too", "short"],
    )
end

@testset "plot_objective split_summary_pages" begin
    f(x) = sum((x .- 1.0).^2)
    x0 = ones(6)
    outdir = mktempdir()

    result = plot_objective(
        f,
        x0;
        outdir = outdir,
        basename = "split",
        resolution = 5,
        output = :summary,
        split_summary_pages = true,
    )

    @test result.summary isa Vector{String}
    @test length(result.summary) == 2
    @test all(isfile, result.summary)
    @test !isfile(joinpath(outdir, "split.pdf"))
end

@testset "run_plot_objective" begin
    f(x) = sum((x .- 1.0).^2)
    x0 = [1.0, 1.0, 1.0]
    outdir = mktempdir()

    manual_result = run_plot_objective(
        f,
        x0;
        outdir = outdir,
        basename = "run_manual",
        output = :summary,
        resolution = 8,
        parallel = :manual,
    )
    @test isfile(manual_result.summary)
    @test manual_result.runtime.parallel == :manual
    @test manual_result.runtime.use_threads == false
    @test manual_result.runtime.blas_threads == 1
    @test manual_result.runtime.auto.strategy == :manual

    if Threads.nthreads() > 1
        threaded_result = run_plot_objective(
            f,
            x0;
            outdir = outdir,
            basename = "run_manual_threads",
            output = :summary,
            resolution = 8,
            parallel = :manual,
            use_threads = true,
            blas_threads = 1,
        )
        @test isfile(threaded_result.summary)
        @test threaded_result.runtime.use_threads == true
        @test threaded_result.runtime.blas_threads == 1
    else
        @test_throws ArgumentError run_plot_objective(
            f,
            x0;
            outdir = outdir,
            basename = "run_manual_threads",
            output = :summary,
            resolution = 8,
            parallel = :manual,
            use_threads = true,
            blas_threads = 1,
        )
    end

    auto_result = run_plot_objective(
        f,
        x0;
        outdir = outdir,
        basename = "run_auto",
        output = :summary,
        resolution = 8,
        parallel = :auto,
        auto_samples_per_thread = 1,
        auto_repeats = 1,
    )
    @test isfile(auto_result.summary)
    @test auto_result.runtime.parallel == :auto
    @test auto_result.runtime.blas_threads >= 1
    @test auto_result.runtime.auto.strategy == :auto
    @test auto_result.runtime.auto.selected in (:threaded_grid, :serial_grid)

    @test_throws ArgumentError run_plot_objective(
        f,
        x0;
        outdir = outdir,
        basename = "run_bad_backend",
        output = :summary,
        resolution = 8,
        backend = :invalid,
    )
    @test_throws ArgumentError run_plot_objective(
        f,
        x0;
        outdir = outdir,
        basename = "run_bad_parallel",
        output = :summary,
        resolution = 8,
        parallel = :invalid,
    )
end

@testset "describe_output_combinations" begin
    function pick(combos; output, one_d, two_d, split_summary_pages)
        idx = findfirst(
            c ->
                c.output == output &&
                    c.one_d == one_d &&
                    c.two_d == two_d &&
                    c.split_summary_pages == split_summary_pages,
            combos,
        )
        @test idx !== nothing
        return combos[idx]
    end

    combos = describe_output_combinations(ones(3); outdir = "out", basename = "base")
    @test length(combos) == 15
    @test all(c -> !(c.one_d == false && c.two_d == false), combos)
    @test !any(c -> c.output == :individual && c.split_summary_pages == true, combos)
    @test all(c -> c.will_error == false, combos)

    c = pick(combos; output = :summary, one_d = true, two_d = true, split_summary_pages = false)
    @test c.nplots == 6
    @test c.npages == 1
    @test c.summary_files == [joinpath("out", "base.pdf")]
    @test c.summary_is_multipage_pdf == false
    @test c.will_error == false

    combos = describe_output_combinations(ones(6); outdir = "out", basename = "base")
    @test length(combos) == 15
    c = pick(combos; output = :summary, one_d = true, two_d = true, split_summary_pages = true)
    @test c.nplots == 21
    @test c.npages == 2
    @test c.summary_files == [joinpath("out", "base_page1.pdf"), joinpath("out", "base_page2.pdf")]
end

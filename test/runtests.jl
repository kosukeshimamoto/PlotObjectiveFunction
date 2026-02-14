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

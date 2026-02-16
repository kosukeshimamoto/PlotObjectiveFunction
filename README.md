# PlotObjectiveFunction

Plot an objective function around an initial/true parameter vector.

## Install

Use Julia `1.9.x` for SLURM runs (recommended: `1.9.2`).

```julia
import Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Usage

```julia
using PlotObjectiveFunction

f(x) = sum((x .- 1.0).^2)

x0 = [1.0, 1.0, 1.0]
plot_objective(f, x0; resolution=50, outdir="plots", range_frac=0.5)
```

This generates:
- A single summary PDF: `objective_plot.pdf`

To also write individual PDFs:

```julia
plot_objective(f, x0; output=:both)
```

To customize parameter names and the summary filename:

```julia
names = ["alpha", "beta", "gamma"]
plot_objective(f, x0; param_names=names, summary_name="my_objective_plot")
```

To write one PDF per page when the summary exceeds 15 plots:

```julia
plot_objective(f, x0; split_summary_pages=true)
```

If x-axis tick labels overlap, reduce the number of x-axis ticks:

```julia
plot_objective(f, x0; xtick_count=3)
```

To change the heatmap colormap and overlay contour lines:

```julia
using Plots
plot_objective(f, x0; colormap=cgrad(:viridis, 256), contour_levels=10)
```

To evaluate objective grids in parallel:

```julia
plot_objective(f, x0; use_threads=true)
```

To run with shared local/SLURM runtime options:

```julia
result = run_plot_objective(
    f,
    x0;
    backend=:auto,          # :auto, :local, :slurm
    parallel=:auto,         # :manual or :auto
    threads_per_task=8,     # optional
    blas_threads=1,         # optional
    use_threads=nothing,    # optional (nothing => auto/manual default)
    resolution=50,
    output=:summary,
)

result.summary
result.runtime
```

To see what files would be written for all combinations of output-related options (dry-run):

```julia
print_output_combinations(x0; outdir="plots", basename="objective_plot")
```

To actually write all combinations into subfolders under `plots/`:

```julia
write_output_combinations(f, x0; outdir="plots/output_combinations", resolution=10)
```

## Local/SLURM Pipeline

The script `examples/quadratic5_demo.jl` works both locally and on SLURM.

Local preview:

```bash
POF_RESOLUTION=12 julia --project=. examples/quadratic5_demo.jl
```

SLURM run (`run_plot.sbatch` example):

```bash
#!/bin/bash
#SBATCH --job-name=plot-objective
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

cd /Users/kosuke/Github/PlotObjectiveFunction
module purge
module load Julia/1.9.2
which julia
julia --version
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GKSwstype=100
export GKS_WSTYPE=100

export POF_BACKEND=slurm
export POF_PARALLEL=auto
export POF_RESOLUTION=50
export POF_OUTDIR=plots/slurm_full

julia --project=. examples/quadratic5_demo.jl
```

Individual files:
- 1D PDFs for each parameter: `objective_plot_1d_p1.pdf`, ...
- 2D PDFs for parameter combinations (i < j): `objective_plot_2d_p1_p2.pdf`, ...

## Notes
- The default range is ±50% around each parameter value.
- `resolution` is the number of points per axis (1D uses `n` points; 2D uses `n^2` evaluations).
- If a parameter value is zero, `zero_range` (default: 1.0) is used instead.
- Set `use_threads=true` to parallelize objective evaluations. Actual speedup requires
  starting Julia with multiple threads (for example, `JULIA_NUM_THREADS=8`).
- On DCC, `Julia/1.11.x` may map to a juliaup launcher in some environments; `Julia/1.9.2`
  is the stable module path for this repository at the moment.
- `run_plot_objective(...; backend=:auto)` automatically switches between local and SLURM modes
  based on `SLURM_JOB_ID`.
- `run_plot_objective(...; parallel=:auto)` benchmarks serial vs threaded grid evaluation and
  chooses the faster option.
- 1D plots include a red vertical line at the `x0` position.
- 2D plots include red cross-lines at `(x0[i], x0[j])`.
- The legend marks the reference point as `True` and the sampled minimum as `Min`.
- Set `show_reference_legend=false` to hide those legend entries.
- Summary pages are A4-sized with 3 columns and 4-5 rows, and plots are ordered
  as 1D first, then 2D.
- In this repository, `plots/` is ignored except for one sample output:
  `plots/quadratic5_demo/objective_plot_quad5.pdf`.

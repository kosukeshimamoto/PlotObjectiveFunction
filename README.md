# PlotObjectiveFunction

Plot an objective function around an initial/true parameter vector.

## Install

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

To see what files would be written for all combinations of output-related options (dry-run):

```julia
print_output_combinations(x0; outdir="plots", basename="objective_plot")
```

To actually write all combinations into subfolders under `plots/`:

```julia
write_output_combinations(f, x0; outdir="plots/output_combinations", resolution=10)
```

Individual files:
- 1D PDFs for each parameter: `objective_plot_1d_p1.pdf`, ...
- 2D PDFs for parameter combinations (i < j): `objective_plot_2d_p1_p2.pdf`, ...

## Notes
- The default range is ±50% around each parameter value.
- `resolution` is the number of points per axis (1D uses `n` points; 2D uses `n^2` evaluations).
- If a parameter value is zero, `zero_range` (default: 1.0) is used instead.
- 1D plots include a red vertical line at the `x0` position.
- 2D plots include red cross-lines at `(x0[i], x0[j])`.
- Summary pages are A4-sized with 3 columns and 4-5 rows, and plots are ordered
  as 1D first, then 2D.

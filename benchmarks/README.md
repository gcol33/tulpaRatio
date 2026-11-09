# tulpaRatio benchmarks

Run these scripts from the package root with tulpaRatio installed in the active R library.

```sh
Rscript benchmarks/test_smoke.R
Rscript benchmarks/bench_hotpath.R
Rscript benchmarks/bench_single_row.R 1 H
Rscript benchmarks/bench_fair.R --seeds 1 --cooldown 0 PG_base
Rscript benchmarks/bench_stan_binomial.R --smoke
```

`bench_single_row.R` selects a model configuration by row number. Its second argument selects the gradient: `H` for hand-coded derivatives or `N` for numerical differences. `bench_fair.R` runs its model or row selections in separate foreground R processes.

`bench_stan_binomial.R` compares binomial fits with Stan through brms. It requires brms and a configured Stan toolchain. `bench_validated.R` and `helpers.R` also require brms for their comparisons. Stan model sources are in `stan/`.

The scripts in `archive/` contain focused sampler checks and timing experiments. Each standalone script loads the installed package. Coordinators use the active worker scripts in `benchmarks/`.

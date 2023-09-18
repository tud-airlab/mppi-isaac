# Benchmark for mppi

If you want to compare mppi to other planners, you can use plannerbenchmark.
As mppi is not part of the standard planners, you must export the following
variable.

```bash
source setup_bench.bash
```

**Note**: if you do not have a license for [ForcesPro](https://www.embotech.com/products/forcespro/overview/), you can exclude the comparison with MPC which is by default included in `run_experiments.sh`. For instance, without ForcesPro you can use:

`runner -c setup/exp.yaml -p setup/mppi.yaml setup/fabric.yaml -n 10 --res-folder results/series --random-goal --random-obst --render`
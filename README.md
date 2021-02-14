# Multi-Robot Path Planning via Integer Programming (SoCG 2021 submission)
This is an implementation project by the [Computational Geometry group](http://www.cs.tufts.edu/research/geometry/) at Tufts University and constitutes part of our submission to the [SoCG 2021 challenge](https://cgshop.ibr.cs.tu-bs.de/competition/cg-shop-2021/#problem-description). Our explorations of other algorithms are [available in a separate repository](https://github.com/jconroy14/coordinated-motion-planning).

This project implements the makespan-minimizing multi-robot motion planning algorithm in Yu and LaValle's "Optimal Multi-Robot Path Planning on Graphs: Complete Algorithms and Effective Heuristics" [(arXiv:1507.03290)](https://arxiv.org/abs/1507.03290). We add additional constraints to handle continuous grid motion, as required by the SoCG challenge.

## Installing
This project depends on Python 3.8, Gurobi 9.1, and a few other dependencies. Gurobi can be [installed with Anaconda](https://www.gurobi.com/gurobi-and-anaconda-for-linux/) and requires a license [(academic licenses are free)](https://www.gurobi.com/academia/academic-program-and-licenses/). Other dependencies can be installed with `pip install -r requirements.txt`.

## Running
The solver works best on small instances (up to 25x25). To generate a solution for the smallest instance, run 
```
python solve_instance.py --db cgshop_2021_instances_01.zip --name small_000_10x10_20_10 --out-file small_000_10x10_20_10.zip  --verbose
```

For larger problems, it may be necessary to use Yu and LaValle's _k_-way split heuristic. We can specify a sequence of _k_-values to try, a time limit per subproblem (the solver terminates on the first successful _k_), and the number of threads to use:
```
python solve_instance.py --db cgshop_2021_instances_01.zip --name small_free_004_20x20_20_80 --n-threads 4 --out-file small_free_004_20x20_20_80.zip -k 2 -k 4 -k 8 --time-limit 600
```

For convenience (e.g. when running on an AWS spot instance), we support uploading solutions to Dropbox. This requires a Dropbox API key with an associated app (see the [Dropbox OAuth documentation](https://www.dropbox.com/lp/developers/reference/oauth-guide)).

```
python solve_instance.py --db cgshop_2021_instances_01.zip --name small_free_004_20x20_20_80 --n-threads 4 --out-file small_free_004_20x20_20_80.zip -k 2 -k 4 -k 8 --time-limit 600 --dropbox-access-token [YOUR TOKEN HERE] --dropbox-out-file /small_free_004_20x20_20_80.zip
```
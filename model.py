"""IP model for (near-)optimal multi-robot path planning.

Heavily based on "Optimal Multi-Robot Path Planning on Graphs:
Complete Algorithms and Effective Heuristics" (Yu and LaValle 2015).
See https://arxiv.org/abs/1507.03290.

We have modified the Yu and LaValle model by adding pseudo-continuous
motion constraints that enforce the SoCG 2021 setting.
"""
import numpy as np
import networkx as nx
from cgshop2021_pyutils import (SolutionStep, Solution, Direction,
                                InstanceBuilder, validate)
import gurobipy as gp
from gurobipy import GRB
from itertools import chain
from tqdm import trange


def dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def lower_bound_T(instance):
    return max(dist(s, t) for s, t in zip(instance.start, instance.target))


def to_networkx(instance, buffer=0):
    shape = instance.description['parameters']['shape']
    graph = nx.grid_graph(
        dim=[shape[0] + (2 * buffer), shape[1] + (2 * buffer)])
    graph = nx.relabel_nodes(graph, {(x, y): (x - buffer, y - buffer)
                                     for x, y in graph.nodes})
    obstacles = []
    for pos, (state, robot) in instance.at_position.items():
        if state in ('start', 'target', 'both'):
            graph.nodes[pos]['robot'] = robot
            graph.nodes[pos]['state'] = state
        elif state == 'obstacle':
            obstacles.append(pos)
    graph.remove_nodes_from(obstacles)

    # Remove inaccessible regions created by obstacles.
    components = list(nx.connected_components(graph))
    if len(components) > 1:
        for component in components:
            robot_count = sum('robot' in graph.nodes[n] for n in component)
            if robot_count == 0:
                graph.remove_nodes_from(component)
    return graph


def k_way_split(instance, graph, buffer, k):
    assert k > 1
    n_robots = instance.number_of_robots
    shape = [
        max(graph.nodes, key=lambda k: k[0])[0] -
        min(graph.nodes, key=lambda k: k[0])[1],
        max(graph.nodes, key=lambda k: k[1])[1] -
        min(graph.nodes, key=lambda k: k[1])[1]
    ]

    paths = {}
    for r in range(n_robots):
        paths[r] = nx.shortest_path(graph, instance.start[r],
                                    instance.target[r])

    paths_by_length = sorted(paths.items(), key=lambda kv: -len(kv[1]))
    plans = -np.ones((k - 1, *shape), dtype=np.int)
    for step in range(k - 1):
        pairs = set((x, y) for x, y in zip(*np.where(plans[step] == -1))
                    if (x - buffer, y - buffer) in graph.nodes)
        for r, path in paths_by_length:
            waypoint = path[((step + 1) * len(path)) // k]
            x = waypoint[0] + buffer
            y = waypoint[1] + buffer
            if plans[step, x, y] == -1:
                plans[step, x, y] = r
                pairs.remove((x, y))
            else:
                # Handle collisions. Heuristics:
                #  (1) [TODO] Try (up to) a constant number of steps back and forward
                #      in the shortest paths.
                #  (2) [Currently implemented] Search for a nearest empty node.
                #      [TODO] update paths after each iteration?
                close_x, close_y = min(
                    pairs, key=lambda p: abs(x - p[0]) + abs(y - p[1]))
                plans[step, close_x, close_y] = r
                pairs.remove((close_x, close_y))

    waypoints = [[instance.start[r]] for r in range(n_robots)]
    for step in range(k - 1):
        pairs = [(x, y) for x, y in zip(*np.where(plans[step] >= 0))]
        for x, y in pairs:
            waypoints[int(plans[step, x, y])].append((x - buffer, y - buffer))
    for r, target in enumerate(instance.target):
        waypoints[r].append(target)

    instances = []
    for step in range(k - 1):
        builder = InstanceBuilder(instance.name + f'_{step}',
                                  instance.description)
        for obstacle in instance.obstacles:
            builder.add_obstacle(obstacle)
        for robot in waypoints:
            builder.add_robot(robot[step], robot[step + 1])
        instances.append(builder.build_instance())
    return instances


class ProblemInstance:
    def __init__(self, instance, T=None, buffer=0):
        if T is None:
            T = lower_bound_T(instance)
        assert T >= lower_bound_T(instance)
        self.instance = instance
        raw_shape = self.instance.description['parameters']['shape']
        self.shape = [raw_shape[0] + (2 * buffer), raw_shape[1] + (2 * buffer)]
        self.unbuffered_shape = raw_shape
        self.T = T
        self.buffer = buffer
        self.graph = to_networkx(self.instance, self.buffer)
        self.flow, self.gadgets = self._to_flow_network()

    def _to_flow_network(self):
        # Make T + 1 copies of each node.
        flow = nx.DiGraph()
        for node in self.graph.nodes:
            for i in range(self.T + 1):
                flow.add_node((*node, i))
        for node in self.graph.nodes:
            for i in range(self.T):
                flow.add_edge((*node, i), (*node, i + 1),
                              function='stationary')

        # Add simplified merge-split gadget to encode adjacency.
        gadgets = []
        for (u, v) in self.graph.edges:
            for i in range(self.T):
                left = ((*u, i), (*v, i + 1))
                right = ((*v, i), (*u, i + 1))
                flow.add_edge(*left, function='ms')
                flow.add_edge(*right, function='ms')
                gadgets.append((left, right))

        for robot, (start, target) in enumerate(
                zip(self.instance.start, self.instance.target)):
            flow.add_edge((*target, self.T), (*start, 0),
                          function='loopback',
                          robot=robot)

        return flow, gadgets

    def solve(self,
              continuous=True,
              time_limit=None,
              obj='makespan',
              start_vals=None,
              verbose=True,
              n_threads=0):
        def _log(s):
            if verbose:
                print(s)

        if verbose:
            _range = trange
        else:
            _range = range

        assert obj in ('makespan', 'energy')
        n_robots = self.instance.number_of_robots
        size = len(self.graph)
        flow = self.flow
        loopback_edges = [
            e for e in flow.edges if flow.edges[e]['function'] == 'loopback'
        ]

        model = gp.Model('robots')
        if not verbose:
            model.setParam('OutputFlag', False)
        if obj == 'makespan':
            model.setParam('Cutoff', self.instance.number_of_robots)
            model.setParam('BestObjStop',
                           self.instance.number_of_robots - 1e-6)
        if time_limit is not None:
            model.setParam('TimeLimit', time_limit)
        if n_threads != 0:
            model.setParam('Threads', n_threads)

        # Each robot r_i has a flow function f_i(e): flow.edges -> {0, 1}.
        # We apply basic reachability analysis here (TODO: improve?)
        multi_edges_by_robot = []
        multi_edges_by_edge = {e: [] for e in self.flow.edges}
        edgeset = set()
        robot_loopback_edges = []
        _log('Generating variables...')
        for robot in _range(n_robots):
            robot_edges = []
            for edge in flow.edges:
                accept = False
                if flow.edges[edge]['function'] == 'loopback':
                    # We implicitly implement the constraint that a robot
                    # cannot be mapped to the loopback edge of another robot.
                    # See p. 7, formula 8, line 2.
                    accept = flow.edges[edge]['robot'] == robot
                    if accept:
                        robot_loopback_edges.append((robot, *edge))
                else:
                    x, y, t = edge[1]
                    accept = dist(self.instance.start[robot], (x, y)) <= t + 1

                if accept:
                    edgeset.add((robot, *edge))
                    robot_edges.append((robot, *edge))
                    multi_edges_by_edge[edge].append((robot, *edge))
            multi_edges_by_robot.append(robot_edges)

        flow_vars = model.addVars(chain(*multi_edges_by_robot),
                                  vtype=GRB.BINARY)
        if start_vals:
            for k, v in start_vals.items():
                if k in flow_vars:
                    flow_vars[k].start = v

        # Constraint: edges have unit capacity over all commodities.
        # (p. 7, formula 8, line 1)
        _log('Adding capacity constraints...')
        model.addConstrs(
            gp.quicksum(flow_vars[re] for re in multi_edges_by_edge[edge]) <= 1
            for edge in flow.edges
            if flow.edges[edge]['function'] == 'stationary')

        # Constraint: flow is conserved (per robot). (p. 7, formula 9)
        _log('Adding flow constraints...')
        for r in range(n_robots):
            model.addConstrs(
                gp.quicksum(flow_vars[(r, *e)] for e in flow.in_edges(v)
                            if (r, *e) in edgeset) == gp.quicksum(
                                flow_vars[(r, *e)] for e in flow.out_edges(v)
                                if (r, *e) in edgeset) for v in flow.nodes)

        # Constraint: head-on collision constraint (p. 9, formula 17)
        _log('Adding head-on collision constraints...')
        model.addConstrs(
            (gp.quicksum(flow_vars[(r, *left)]
                         for r in range(n_robots) if (r, *left) in edgeset) +
             gp.quicksum(flow_vars[(r, *right)] for r in range(n_robots) if
                         (r, *right) in edgeset)) <= 1
            for left, right in self.gadgets)

        # Constraint: meet collision constraint (p. 9, formula 18)
        _log('Adding meet collision constraints...')
        model.addConstrs(
            gp.quicksum(flow_vars[(r, *e)] for r in range(n_robots)
                        for e in flow.out_edges(v) if (r, *e) in edgeset) <= 1
            for v in flow.nodes)

        # Constraint (NEW): right-angle collision constraints
        # for pseudo-continuous motion.
        if continuous:
            _log('Adding pseudo-continuous motion constraints...')
            for x, y in self.graph.nodes:
                left = (x - 1, y) if (x - 1, y) in self.graph.nodes else None
                right = (x + 1, y) if (x + 1, y) in self.graph.nodes else None
                up = (x, y - 1) if (x, y - 1) in self.graph.nodes else None
                down = (x, y + 1) if (x, y + 1) in self.graph.nodes else None
                center = (x, y)
                problematic_pairs = [  # in (t, t + 1) order
                    pair for pair in ((up, left), (up, right), (down, left),
                                      (down, right), (left, up), (left, down),
                                      (right, up), (right, down))
                    if pair[0] is not None and pair[1] is not None
                ]

                for t in range(self.T):
                    model.addConstrs(
                        (gp.quicksum(flow_vars[(r, (*center, t),
                                                (*start, t + 1))]
                                     for r in range(n_robots) if
                                     (r, (*center, t),
                                      (*start, t + 1)) in edgeset) +
                         gp.quicksum(flow_vars[(r, (*end, t),
                                                (*center, t + 1))]
                                     for r in range(n_robots) if
                                     (r, (*end, t),
                                      (*center, t + 1)) in edgeset)) <= 1
                        for start, end in problematic_pairs)

        if obj == 'makespan':
            # Objective: maximize flow over loopback edges---that is, make
            # sure as many robots move from their source to their target
            # as possible. A solution is valid only when *every* robot
            # is sent across its loopback edge---that is, the objective
            # is exactly equal to `n_robots`.
            model.setObjective(
                gp.quicksum(flow_vars[e] for e in robot_loopback_edges),
                GRB.MAXIMIZE)
        elif obj == 'energy':
            # Objective: minimize overall motion. We enforce solution validity
            # with extra constraints.
            model.addConstrs(flow_vars[e] == 1 for e in robot_loopback_edges)
            model.setObjective(
                gp.quicksum(flow_vars[e] for e in edgeset
                            if e[1][:2] != e[2][:2]), GRB.MINIMIZE)

        model.optimize()
        if model.status == GRB.USER_OBJ_LIMIT or model.status == GRB.OPTIMAL:
            robot_abs_pos = {r: [] for r in range(n_robots)}
            for (robot, start, end), var in flow_vars.items():
                if abs(var.x - 1) < 1e-6:
                    robot_abs_pos[robot].append(end)
            for r, steps in robot_abs_pos.items():
                robot_abs_pos[r] = sorted(steps, key=lambda v: v[2])

            robot_rel_pos = {r: {} for r in range(n_robots)}
            for r in range(n_robots):
                for (xs, ys, _), (xt, yt, tt) in zip(robot_abs_pos[r][:-1],
                                                     robot_abs_pos[r][1:]):
                    if yt - ys == 1:
                        robot_rel_pos[r][tt] = 'N'
                    elif ys - yt == 1:
                        robot_rel_pos[r][tt] = 'S'
                    elif xt - xs == 1:
                        robot_rel_pos[r][tt] = 'E'
                    elif xs - xt == 1:
                        robot_rel_pos[r][tt] = 'W'

            n_steps = max(
                max(r.keys()) for r in robot_rel_pos.values() if len(r) > 0)
            at_timestep = [{} for _ in range(n_steps)]
            for t in range(1, n_steps + 1):
                for r, steps in robot_rel_pos.items():
                    if t in steps:
                        at_timestep[t - 1][r] = steps[t]

            directions = {
                'N': Direction.NORTH,
                'W': Direction.WEST,
                'E': Direction.EAST,
                'S': Direction.SOUTH
            }
            sol = Solution(self.instance)
            for step in at_timestep:
                sol_step = SolutionStep()
                for k, d in step.items():
                    sol_step[k] = directions[d]
                sol.add_step(sol_step)
            validate(sol)
            return sol

        # If the solver fails (usually due to a time limit or invalid T),
        # reuse whatever partial solution we found as a warm start.
        # TODO: this doesn't actually work, at least when `Cutoff` is enabled.
        # (Is there another setting we can tweak?)
        return {
            'status': model.status,
            'vals': {k: v.x
                     for k, v in flow_vars.items() if hasattr(v, 'x')}
        }


def optimize_makespan(instance,
                      buffer,
                      time_limit,
                      step_size,
                      step_limit,
                      polish=True,
                      verbose=False,
                      n_threads=0):
    lb = lower_bound_T(instance)
    right_T = lb
    problem = ProblemInstance(instance, buffer=buffer, T=right_T)
    right_sol = problem.solve(time_limit=time_limit,
                              verbose=verbose,
                              n_threads=n_threads)
    left_T = None
    left_sol = None

    while not isinstance(right_sol, Solution) and right_T <= step_limit * lb:
        left_T = right_T
        left_sol = right_sol
        right_T *= step_size
        if verbose:
            print('[exponential search] solving with makespan of',
                  int(round(right_T)))
        problem = ProblemInstance(instance,
                                  buffer=buffer,
                                  T=int(round(right_T)))
        right_sol = problem.solve(time_limit=time_limit,
                                  start_vals=left_sol['vals'],
                                  verbose=verbose,
                                  n_threads=n_threads)

        if isinstance(right_sol,
                      dict) and right_sol.get('status') == GRB.TIME_LIMIT:
            # Trying a larger T will almost certainly take
            # even longer (model blowup).
            return None

    if isinstance(right_sol, Solution):
        if polish and left_T is not None:
            # Binary search for the optimal T.
            left_T = int(round(left_T))
            right_T = int(round(right_T))
            while left_T < right_T:
                mid_T = int((left_T + right_T) / 2)
                if verbose:
                    print('[binary search] solving with makespan of', mid_T)
                problem = ProblemInstance(instance, buffer=buffer, T=mid_T)
                mid_sol = problem.solve(time_limit=time_limit,
                                        start_vals=left_sol['vals'],
                                        verbose=verbose,
                                        n_threads=n_threads)
                if isinstance(mid_sol, Solution):
                    # We can afford to make the makespan smaller.
                    right_T = mid_T
                    right_sol = mid_sol
                else:
                    left_T = mid_T + 1
                    left_sol = mid_sol
        return right_sol


def optimize_makespan_k_ways(instance,
                             buffer,
                             k,
                             time_limit=180,
                             step_size=1.25,
                             step_limit=1.25**4,
                             polish=True,
                             verbose=False,
                             n_threads=0):
    if k == 1:
        return optimize_makespan(instance,
                                 buffer=buffer,
                                 time_limit=time_limit,
                                 step_size=step_size,
                                 step_limit=step_limit,
                                 polish=polish,
                                 verbose=verbose,
                                 n_threads=n_threads)

    graph = to_networkx(instance, buffer=buffer)
    subinstances = k_way_split(instance, graph, buffer=buffer, k=k)
    sol = Solution(instance)
    for idx, subinstance in enumerate(subinstances):
        print('[k-way] solving subinstance', idx)
        subsol = optimize_makespan(subinstance,
                                   buffer=buffer,
                                   time_limit=time_limit,
                                   step_size=step_size,
                                   step_limit=step_limit,
                                   polish=polish,
                                   verbose=verbose,
                                   n_threads=n_threads)
        if isinstance(subsol, Solution):
            for step in subsol.steps:
                sol.add_step(step)
        else:
            return None  # subinstance failed
    validate(sol)
    return sol

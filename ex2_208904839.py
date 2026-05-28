"""
AI Course - Assignment 2: Stochastic Multi-Elevator Passenger
Bar-Ilan CS 89-570

AI disclosure:
  Used: Claude Code for brainstorming controller strategies, discussing the
  MDP formulation, and discussing admissibility/consistency of the inner A*
  heuristic. The code was written by Claude. I (Shaked) directed the design
  through discussion, verified correctness against the specification, ran the
  local checker, and validated the implementation.

Architecture: a hybrid policy.
  - When the reachable state space is small enough to solve within the time
    budget, an EXACT finite-horizon MDP policy is computed by backward
    induction (value iteration). This maximises expected total reward and
    handles defective elevators / reset timing optimally.
  - For larger problems, the controller falls back to cost-shaped A* replan
    with lazy plan following.
"""

import ext_elev
import heapq
import time
from collections import deque

id = ["208904839"]

# Cache of solved MDP policies, keyed by a signature derived purely from
# the documented GameAPI getters. The optimal policy depends only on the
# problem structure (identical across seeds), so this memoises our own
# value-iteration result. Holds the most recent problem only.
_MDP_CACHE = {}


# --------------------------------------------------------------------------- #
# Internal deterministic planning state (used by inner A*).                   #
# Reuses the Assignment-1 encoding:                                           #
#   elev_floors: tuple[int]  – floor of each elevator in sorted-ID order      #
#   person_locs: tuple[int]  – for each person in sorted-ID order:            #
#                                  loc >= 0 -> standing on floor `loc`        #
#                                  loc <  0 -> inside elevator with idx       #
#                                                eidx = -loc - 1              #
# --------------------------------------------------------------------------- #
class _PlanState:
    __slots__ = ("elev_floors", "person_locs", "_hash")

    def __init__(self, elev_floors, person_locs):
        self.elev_floors = elev_floors
        self.person_locs = person_locs
        self._hash = hash((elev_floors, person_locs))

    def __eq__(self, other):
        return (self.elev_floors == other.elev_floors
                and self.person_locs == other.person_locs)

    def __hash__(self):
        return self._hash


class Controller:
    """Stochastic multi-elevator controller.

    Architecture: cost-shaped FF-replan with lazy plan following.
      1. In __init__: read static info, precompute analyses, run A* once.
      2. choose_next_action: if last action succeeded (state matches the
         expected next-state), advance the cached plan; otherwise replan.
      3. Before returning an action, compare against RESET via a
         lightweight Q estimate.
    """

    # ----------------------------------------------------------------- #
    # Construction & precomputation                                     #
    # ----------------------------------------------------------------- #
    def __init__(self, game: ext_elev.GameAPI):
        self.game = game

        # ---- pull static info via the GameAPI ------------------------ #
        self.reachable = game.get_reachable()          # eid -> frozenset
        self.capacities = game.get_capacities()        # eid -> int
        initial_state = game.get_initial_state()
        elevators_t0, persons_t0, _ = initial_state

        self.elevator_ids = sorted(self.reachable.keys())
        self.person_ids = sorted(pid for pid, _ in persons_t0)
        self.eidx_of = {eid: i for i, eid in enumerate(self.elevator_ids)}
        self.pidx_of = {pid: i for i, pid in enumerate(self.person_ids)}

        # static per-elevator
        self.elev_reachable = [
            self.reachable[eid] for eid in self.elevator_ids
        ]
        self.elev_capacity = [
            self.capacities[eid] for eid in self.elevator_ids
        ]
        self.elev_prob = [
            game.get_elevator_action_prob(eid)
            for eid in self.elevator_ids
        ]
        # cost-shaped action cost for each elevator's MOVE.
        # Full 1/p is admissibility-tight but, under our serial-action
        # model, overcharges broken elevators (failed MOVEs let us
        # interleave useful work elsewhere). Softened with alpha=0.5:
        #   cost = 1 + 0.5 * (1/p - 1)
        # For p=0.95 -> 1.026 (was 1.053). For p=0.30 -> 2.167 (was 3.333).
        self.elev_move_cost = [
            1.0 + 0.5 * (1.0 / p - 1.0) for p in self.elev_prob
        ]

        # static per-person
        self.person_weight = [
            game.get_person_weight(pid) for pid in self.person_ids
        ]
        self.person_goal = [
            game.get_person_goal(pid) for pid in self.person_ids
        ]
        self.person_prob = [
            game.get_person_action_prob(pid) for pid in self.person_ids
        ]
        # Same softening for ENTER/EXIT costs.
        self.person_action_cost = [
            1.0 + 0.5 * (1.0 / p - 1.0) for p in self.person_prob
        ]
        # mean reward per person (uniform over the reward list)
        self.person_mean_reward = []
        for pid in self.person_ids:
            rewards = game.get_person_reward(pid)
            self.person_mean_reward.append(sum(rewards) / len(rewards))

        # global problem info
        self.goal_reward = game.get_goal_reward()
        self.max_steps = game.get_max_steps()

        # goal-state encoding (used by goal_test)
        self.goal_locs_tuple = tuple(self.person_goal)

        # ---- precomputations (BFS-based, like A1) -------------------- #
        self._compute_dist_to_goal()
        self._compute_transitive_reach()
        self._compute_useful_exit_floors()
        self._compute_min_stints_in_elev()
        self._compute_elev_overlap()
        self._compute_transfer_floors()

        # Pick the BEST strategy: deliver-full vs single-person reset-loop.
        # Each strategy = (target_subset, plan, reward_per_loop, cost_per_loop).
        self._initial_state = initial_state
        self._pick_best_strategy()

        # Start executing the chosen strategy's plan from the initial state.
        self.plan = self._a_star_for_subset(initial_state, self.target_subset)
        self.plan_index = 0
        if self.plan:
            self._expected_state = self._simulate_success(
                initial_state, self.plan[0]
            )
        else:
            self._expected_state = None

        # When the reachable state space is small enough, replace the
        # FF-replan heuristic with an exact finite-horizon MDP policy
        # (backward-induction value iteration). This is optimal for the
        # expected total reward and handles broken elevators / reset
        # timing without cost-shaping. Large problems keep the A* path.
        self._mdp_V = None
        self._mdp_state_to_idx = None
        self._try_build_mdp()

    # ----------------------------------------------------------------- #
    # Required API                                                      #
    # ----------------------------------------------------------------- #
    def choose_next_action(self, state):
        if self._mdp_V is not None:
            mdp_action = self._mdp_choose(state)
            if mdp_action is not None:
                return mdp_action

        elevators_t, persons_t, _total = state

        # If our target subset is fully delivered in the current state,
        # there's nothing more to do this "episode" → RESET to fetch the
        # high-reward subset again. (Key for rl-style cases.)
        undelivered_ids = {pid for pid, _ in persons_t}
        if not (self.target_subset & undelivered_ids):
            self.plan = None
            self.plan_index = 0
            self._expected_state = self.game.get_initial_state()
            return "RESET"

        # Fast path: state matches what we expected → advance plan
        if (self._expected_state is not None
                and self._states_equal(state, self._expected_state)
                and self.plan
                and self.plan_index < len(self.plan)):
            action = self.plan[self.plan_index]
            self.plan_index += 1
        else:
            # Replan from current state, restricted to target subset
            self.plan = self._a_star_for_subset(state, self.target_subset)
            self.plan_index = 0
            if self.plan and len(self.plan) > 0:
                action = self.plan[0]
                self.plan_index = 1
            else:
                # No plan reachable → RESET as a safe default
                self.plan = None
                self.plan_index = 0
                self._expected_state = self.game.get_initial_state()
                return "RESET"

        # Defensive legality check
        if not self._is_legal(state, action):
            self.plan = None
            self.plan_index = 0
            self._expected_state = self.game.get_initial_state()
            return "RESET"

        # Update expected-next-state
        self._expected_state = self._simulate_success(state, action)
        return action

    # ----------------------------------------------------------------- #
    # Exact MDP policy (finite-horizon value iteration)                 #
    #                                                                   #
    # Used only when the reachable state space is small enough to solve #
    # exactly within the time budget. Internal state encoding:          #
    #   elev_floors : tuple[int]  – floor of each elevator              #
    #   person_locs : tuple[int]  – per person:                         #
    #         loc >= 0            -> standing on floor `loc`             #
    #         _MDP_DELIVERED       -> delivered (removed from episode)   #
    #         else (negative)      -> inside elevator (-loc - 1)         #
    # ----------------------------------------------------------------- #
    _MDP_DELIVERED = 1_000_000

    def _mdp_encode(self, a2_state):
        elevators_t, persons_t, _ = a2_state
        ef = [0] * len(self.elevator_ids)
        for e, fl, _w in elevators_t:
            ef[self.eidx_of[e]] = fl
        pl = [self._MDP_DELIVERED] * len(self.person_ids)
        for pid, loc in persons_t:
            idx = self.pidx_of[pid]
            if loc[0] == 'floor':
                pl[idx] = loc[1]
            else:
                pl[idx] = -self.eidx_of[loc[1]] - 1
        return (tuple(ef), tuple(pl))

    def _mdp_legal_actions(self, state):
        ef, pl = state
        DELIV = self._MDP_DELIVERED
        n_elev = len(self.elevator_ids)
        n_pers = len(self.person_ids)
        load = [0] * n_elev
        for i, loc in enumerate(pl):
            if loc != DELIV and loc < 0:
                load[-loc - 1] += self.person_weight[i]
        actions = []
        for eidx in range(n_elev):
            cur = ef[eidx]
            for f in self.elev_reachable[eidx]:
                if f != cur:
                    actions.append(('MOVE', eidx, f))
        for pidx in range(n_pers):
            loc = pl[pidx]
            if loc == DELIV or loc < 0:
                continue
            for eidx in range(n_elev):
                if (ef[eidx] == loc
                        and load[eidx] + self.person_weight[pidx]
                        <= self.elev_capacity[eidx]):
                    actions.append(('ENTER', pidx, eidx))
        for pidx in range(n_pers):
            loc = pl[pidx]
            if loc != DELIV and loc < 0:
                actions.append(('EXIT', pidx, -loc - 1))
        actions.append(('RESET',))
        return actions

    def _mdp_outcomes(self, state, action):
        ef, pl = state
        DELIV = self._MDP_DELIVERED
        kind = action[0]
        if kind == 'RESET':
            return [(1.0, self._mdp_init_state, 0.0)]
        if kind == 'MOVE':
            _, eidx, target = action
            p = self.elev_prob[eidx]
            ef_s = list(ef); ef_s[eidx] = target
            outs = [(p, (tuple(ef_s), pl), 0.0)]
            others = [f for f in self.elev_reachable[eidx] if f != target]
            if others:
                pf = (1.0 - p) / len(others)
                for f in others:
                    ef_f = list(ef); ef_f[eidx] = f
                    outs.append((pf, (tuple(ef_f), pl), 0.0))
            return outs
        if kind == 'ENTER':
            _, pidx, eidx = action
            q = self.person_prob[pidx]
            pl_s = list(pl); pl_s[pidx] = -eidx - 1
            return [(q, (ef, tuple(pl_s)), 0.0), (1.0 - q, state, 0.0)]
        if kind == 'EXIT':
            _, pidx, eidx = action
            q = self.person_prob[pidx]
            floor = ef[eidx]
            if floor == self.person_goal[pidx]:
                pl_s = list(pl); pl_s[pidx] = DELIV
                reward = self.person_mean_reward[pidx]
                if all(x == DELIV for x in pl_s):
                    reward += self.goal_reward
                    nxt = self._mdp_init_state
                else:
                    nxt = (ef, tuple(pl_s))
                return [(q, nxt, reward), (1.0 - q, state, 0.0)]
            pl_s = list(pl); pl_s[pidx] = floor
            return [(q, (ef, tuple(pl_s)), 0.0), (1.0 - q, state, 0.0)]
        return [(1.0, state, 0.0)]

    def _problem_signature(self):
        return (
            tuple(self.elevator_ids),
            tuple(tuple(sorted(r)) for r in self.elev_reachable),
            tuple(self.elev_capacity),
            tuple(self.elev_prob),
            tuple(self.person_ids),
            tuple(self.person_weight),
            tuple(self.person_goal),
            tuple(self.person_prob),
            tuple(self.person_mean_reward),
            self.goal_reward,
            self.max_steps,
            self._mdp_init_state,
        )

    def _try_build_mdp(self):
        self._mdp_init_state = self._mdp_encode(self._initial_state)
        horizon = self.max_steps
        sig = self._problem_signature()
        cached = _MDP_CACHE.get(sig, "MISS")
        if cached is False:
            return  # known too-large → A* path
        if cached != "MISS":
            self._mdp_V, self._mdp_state_to_idx = cached
            return
        # Wall-clock safety: abort to A* if the build approaches the
        # per-seed time limit (20 + 0.5*horizon), so a slow machine can
        # never blow the budget. Use a conservative 0.5 fraction.
        build_start = time.perf_counter()
        deadline = build_start + 0.5 * (20.0 + 0.5 * horizon)
        # BFS over reachable states, capped so huge problems abort fast.
        cap = 40_000
        seen = {self._mdp_init_state}
        frontier = deque([self._mdp_init_state])
        while frontier:
            s = frontier.popleft()
            for a in self._mdp_legal_actions(s):
                for _p, ns, _r in self._mdp_outcomes(s, a):
                    if ns not in seen:
                        seen.add(ns)
                        if len(seen) > cap:
                            _MDP_CACHE.clear()
                            _MDP_CACHE[sig] = False
                            return  # too large → keep A* path
                        frontier.append(ns)
        # Time-budget gate: estimated build vs (20 + 0.5*horizon) limit.
        if len(seen) * horizon > 35_000 * (20 + 0.5 * horizon):
            _MDP_CACHE.clear()
            _MDP_CACHE[sig] = False
            return
        if time.perf_counter() > deadline:
            _MDP_CACHE.clear()
            _MDP_CACHE[sig] = False
            return
        states = list(seen)
        state_to_idx = {s: i for i, s in enumerate(states)}
        n = len(states)
        # Precompute transitions as (imm_reward, [(next_idx, prob), ...]).
        trans = []
        for s in states:
            acts = self._mdp_legal_actions(s)
            per_state = []
            for a in acts:
                imm = 0.0
                nexts = []
                for prob, ns, r in self._mdp_outcomes(s, a):
                    imm += prob * r
                    nexts.append((state_to_idx[ns], prob))
                per_state.append((imm, nexts))
            trans.append(per_state)
        # Backward induction. V_prev = V at t-1; build V layers 1..horizon.
        V_prev = [0.0] * n
        V_layers = [V_prev]
        for _t in range(1, horizon + 1):
            if time.perf_counter() > deadline:
                _MDP_CACHE.clear()
                _MDP_CACHE[sig] = False
                return  # build too slow on this machine → A* path
            V_cur = [0.0] * n
            for idx in range(n):
                best = -1.0
                for imm, nexts in trans[idx]:
                    q = imm
                    for nidx, prob in nexts:
                        q += prob * V_prev[nidx]
                    if q > best:
                        best = q
                V_cur[idx] = best
            V_layers.append(V_cur)
            V_prev = V_cur
        self._mdp_V = V_layers
        self._mdp_state_to_idx = state_to_idx
        _MDP_CACHE.clear()
        _MDP_CACHE[sig] = (V_layers, state_to_idx)

    def _mdp_choose(self, a2_state):
        remaining = self.max_steps - self.game.get_current_steps()
        if remaining <= 0:
            return None
        s = self._mdp_encode(a2_state)
        idx = self._mdp_state_to_idx.get(s)
        if idx is None:
            return None  # off-policy state → defer to A*
        V_next = self._mdp_V[remaining - 1]
        best = -1.0
        best_action = ('RESET',)
        for a in self._mdp_legal_actions(s):
            q = 0.0
            for prob, ns, r in self._mdp_outcomes(s, a):
                nidx = self._mdp_state_to_idx.get(ns)
                vn = V_next[nidx] if nidx is not None else 0.0
                q += prob * (r + vn)
            if q > best:
                best = q
                best_action = a
        return self._mdp_action_str(best_action)

    def _mdp_action_str(self, action):
        kind = action[0]
        if kind == 'RESET':
            return 'RESET'
        if kind == 'MOVE':
            _, eidx, f = action
            return f"MOVE{{{self.elevator_ids[eidx]},{f}}}"
        if kind == 'ENTER':
            _, pidx, eidx = action
            return f"ENTER{{{self.person_ids[pidx]},{self.elevator_ids[eidx]}}}"
        _, pidx, eidx = action
        return f"EXIT{{{self.person_ids[pidx]},{self.elevator_ids[eidx]}}}"

    # ----------------------------------------------------------------- #
    # Precomputations (BFS-based, mirroring Assignment 1)               #
    # ----------------------------------------------------------------- #
    def _compute_dist_to_goal(self):
        """For each goal floor g, BFS in the floor-connectivity graph."""
        relevant_floors = set()
        for reach in self.elev_reachable:
            relevant_floors |= reach
        relevant_floors.update(self.person_goal)

        adj = {f: set() for f in relevant_floors}
        for reach in self.elev_reachable:
            reach_list = list(reach)
            for i in range(len(reach_list)):
                a = reach_list[i]
                for j in range(i + 1, len(reach_list)):
                    b = reach_list[j]
                    adj[a].add(b)
                    adj[b].add(a)

        goal_set = set(self.person_goal)
        self.dist_to_goal = {}
        for g in goal_set:
            if g not in adj:
                self.dist_to_goal[g] = {g: 0}
                continue
            dist = {g: 0}
            q = deque([g])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
            self.dist_to_goal[g] = dist

    def _compute_transitive_reach(self):
        n = len(self.elevator_ids)
        e_adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if self.elev_reachable[i] & self.elev_reachable[j]:
                    e_adj[i].add(j)
                    e_adj[j].add(i)

        self.elev_transitive = []
        for i in range(n):
            visited = {i}
            q = deque([i])
            union = set(self.elev_reachable[i])
            while q:
                u = q.popleft()
                for v in e_adj[u]:
                    if v not in visited:
                        visited.add(v)
                        union |= self.elev_reachable[v]
                        q.append(v)
            self.elev_transitive.append(frozenset(union))

    def _compute_useful_exit_floors(self):
        self.useful_exit = []
        for pidx in range(len(self.person_ids)):
            g = self.person_goal[pidx]
            useful = {g}
            for eidx2 in range(len(self.elevator_ids)):
                if g in self.elev_transitive[eidx2]:
                    useful |= self.elev_reachable[eidx2]
            self.useful_exit.append(frozenset(useful))

    def _compute_min_stints_in_elev(self):
        n = len(self.elevator_ids)
        INF = 10 ** 9
        self.min_stints_in_elev = [dict() for _ in range(n)]
        goal_set = set(self.person_goal)
        for eidx in range(n):
            reach = self.elev_reachable[eidx]
            for g in goal_set:
                if g in reach:
                    self.min_stints_in_elev[eidx][g] = 1
                else:
                    g_dist = self.dist_to_goal[g]
                    best = INF
                    for fp in reach:
                        d = g_dist.get(fp, INF)
                        if d < best:
                            best = d
                    self.min_stints_in_elev[eidx][g] = (
                        1 + best if best < INF else INF
                    )

    def _compute_elev_overlap(self):
        n = len(self.elevator_ids)
        self.elev_overlap = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                self.elev_overlap[(i, j)] = (
                    self.elev_reachable[i] & self.elev_reachable[j]
                )

    def _compute_transfer_floors(self):
        n = len(self.elevator_ids)
        self.transfer_floors = [dict() for _ in range(n)]
        goal_set = set(self.person_goal)
        for eidx in range(n):
            for g in goal_set:
                if g in self.elev_reachable[eidx]:
                    continue
                acc = set()
                for other in range(n):
                    if other == eidx:
                        continue
                    if g in self.elev_transitive[other]:
                        acc |= self.elev_overlap[(eidx, other)]
                self.transfer_floors[eidx][g] = frozenset(acc)

    # ----------------------------------------------------------------- #
    # Strategy picking (full delivery vs single-person reset-loop)      #
    # ----------------------------------------------------------------- #
    def _pick_best_strategy(self):
        """Pick the target-subset with highest reward-per-step ratio.
        Considers: full-delivery (auto-reset on completion via
        goal_reward) and partial-subset reset-loops (we RESET ourselves
        after delivering the subset). Enumerates singletons, pairs, and
        the full set — adequate for problems with up to ~6 persons."""
        import itertools as _it

        n = len(self.person_ids)
        candidate_subsets = [frozenset(self.person_ids)]
        # Enumerate all non-empty subsets up to size 3 (cheap for n ≤ 5).
        # For larger n, only consider singletons + pairs + full.
        max_k = 3 if n <= 5 else 2
        for k in range(1, min(max_k, n) + 1):
            for combo in _it.combinations(self.person_ids, k):
                candidate_subsets.append(frozenset(combo))

        best_per_step = -1.0
        best_subset = frozenset(self.person_ids)
        best_value_per_episode = sum(self.person_mean_reward) + self.goal_reward
        best_steps_per_episode = 1.0

        for subset in candidate_subsets:
            plan = self._a_star_for_subset(self._initial_state, subset)
            if plan is None:
                continue
            cost = self._plan_cost_estimate(
                self._a2_to_subset_plan_state(self._initial_state, subset),
                plan,
            )
            cost = max(1.0, cost)
            is_full = (subset == frozenset(self.person_ids))
            if is_full:
                # Episode auto-resets after delivering everyone; no
                # manual RESET needed.
                reward = sum(self.person_mean_reward) + self.goal_reward
                eff_cost = cost
            else:
                # We RESET ourselves after delivering the subset; +1 step
                reward = sum(self.person_mean_reward[self.pidx_of[pid]]
                             for pid in subset)
                eff_cost = cost + 1.0
            per_step = reward / eff_cost
            if per_step > best_per_step:
                best_per_step = per_step
                best_subset = subset
                best_value_per_episode = reward
                best_steps_per_episode = eff_cost

        self.target_subset = best_subset
        self.value_per_episode = best_value_per_episode
        self.expected_episode_steps = best_steps_per_episode

    # ----------------------------------------------------------------- #
    # A* tailored to deliver a specific subset of persons               #
    # (other persons are virtually placed at their goal, so the planner #
    #  ignores them for the goal test but still respects their weight   #
    #  inside elevators if they happen to be there.)                    #
    # ----------------------------------------------------------------- #
    def _a_star_for_subset(self, a2_state, subset):
        plan_state = self._a2_to_subset_plan_state(a2_state, subset)
        return self._a_star(plan_state)

    def _a2_to_subset_plan_state(self, a2_state, subset):
        """Like _a2_to_plan_state but only persons in `subset` are
        tracked at their real location; everyone else is marked at goal
        (so they appear 'delivered' to the planner)."""
        elevators_t, persons_t, _ = a2_state
        elev_floors = [0] * len(self.elevator_ids)
        for eid, fl, _w in elevators_t:
            elev_floors[self.eidx_of[eid]] = fl

        person_locs = list(self.person_goal)  # default: at goal
        persons_lookup = {pid: loc for pid, loc in persons_t}
        for pid in subset:
            if pid not in persons_lookup:
                continue  # already delivered in this episode
            loc = persons_lookup[pid]
            idx = self.pidx_of[pid]
            if loc[0] == 'floor':
                person_locs[idx] = loc[1]
            else:
                person_locs[idx] = -self.eidx_of[loc[1]] - 1
        return _PlanState(tuple(elev_floors), tuple(person_locs))

    # ----------------------------------------------------------------- #
    # State conversion (A2 state -> internal _PlanState)                #
    # ----------------------------------------------------------------- #
    def _a2_to_plan_state(self, a2_state):
        elevators_t, persons_t, _ = a2_state
        # elevator floors indexed by sorted eidx
        elev_floors = [0] * len(self.elevator_ids)
        for eid, floor, _w in elevators_t:
            elev_floors[self.eidx_of[eid]] = floor
        # person locations: undelivered -> their location; delivered -> at goal
        # (default: at goal; overwrite for undelivered persons)
        person_locs = list(self.person_goal)
        for pid, loc in persons_t:
            idx = self.pidx_of[pid]
            if loc[0] == 'floor':
                person_locs[idx] = loc[1]
            else:  # ('in', eid)
                person_locs[idx] = -self.eidx_of[loc[1]] - 1
        return _PlanState(tuple(elev_floors), tuple(person_locs))

    def _states_equal(self, a2_state, expected_a2_state):
        """Compare two A2 states for equality (deep structure)."""
        if expected_a2_state is None:
            return False
        return a2_state == expected_a2_state

    # ----------------------------------------------------------------- #
    # Successor generation (deterministic, cost-shaped)                 #
    # ----------------------------------------------------------------- #
    def _is_goal_state(self, plan_state):
        return plan_state.person_locs == self.goal_locs_tuple

    def _successors(self, plan_state):
        """Yield (action_str, next_plan_state, cost) tuples.

        Mirrors A1's pruning: only generates moves to floors that are
        pickup floors / delivery targets / transfer floors.
        """
        elev_floors = plan_state.elev_floors
        person_locs = plan_state.person_locs

        n_elev = len(self.elevator_ids)
        n_pers = len(self.person_ids)

        # classify persons
        on_floor_persons = []
        in_elev_persons = [[] for _ in range(n_elev)]
        elev_load = [0] * n_elev
        for pidx in range(n_pers):
            loc = person_locs[pidx]
            g = self.person_goal[pidx]
            if loc < 0:
                eidx = -loc - 1
                in_elev_persons[eidx].append((pidx, g))
                elev_load[eidx] += self.person_weight[pidx]
            else:
                on_floor_persons.append((pidx, loc, g))

        # elev_must_exit: force EXIT before MOVE
        elev_must_exit = [False] * n_elev
        for eidx in range(n_elev):
            ef = elev_floors[eidx]
            for _, g in in_elev_persons[eidx]:
                if g == ef:
                    elev_must_exit[eidx] = True
                    break

        successors = []

        # MOVE actions
        for eidx in range(n_elev):
            if elev_must_exit[eidx]:
                continue
            cur_floor = elev_floors[eidx]
            reach = self.elev_reachable[eidx]
            elev_trans = self.elev_transitive[eidx]
            move_cost = self.elev_move_cost[eidx]

            candidates = set()
            # pickup with R1
            for _, loc, g in on_floor_persons:
                if loc != g and loc in reach and g in elev_trans:
                    candidates.add(loc)
            # delivery / transfer for own passengers
            tf_for_e = self.transfer_floors[eidx]
            for _, g in in_elev_persons[eidx]:
                if g in reach:
                    candidates.add(g)
                else:
                    candidates |= tf_for_e[g]

            candidates.discard(cur_floor)

            eid = self.elevator_ids[eidx]
            prefix = elev_floors[:eidx]
            suffix = elev_floors[eidx + 1:]
            for target in candidates:
                new_floors = prefix + (target,) + suffix
                new_state = _PlanState(new_floors, person_locs)
                successors.append(
                    (f"MOVE{{{eid},{target}}}", new_state, move_cost)
                )

        # ENTER actions
        for pidx, loc, g in on_floor_persons:
            if loc == g:
                continue
            pid = self.person_ids[pidx]
            w = self.person_weight[pidx]
            pre = person_locs[:pidx]
            post = person_locs[pidx + 1:]
            enter_cost = self.person_action_cost[pidx]
            for eidx in range(n_elev):
                if elev_floors[eidx] != loc:
                    continue
                if elev_must_exit[eidx]:
                    continue
                if g not in self.elev_transitive[eidx]:
                    continue
                if elev_load[eidx] + w > self.elev_capacity[eidx]:
                    continue
                encoded = -eidx - 1
                new_locs = pre + (encoded,) + post
                new_state = _PlanState(elev_floors, new_locs)
                successors.append(
                    (f"ENTER{{{pid},{self.elevator_ids[eidx]}}}",
                     new_state, enter_cost)
                )

        # EXIT actions
        for eidx in range(n_elev):
            ef = elev_floors[eidx]
            eid = self.elevator_ids[eidx]
            for pidx, g in in_elev_persons[eidx]:
                if ef not in self.useful_exit[pidx]:
                    continue
                exit_cost = self.person_action_cost[pidx]
                new_locs = (
                    person_locs[:pidx] + (ef,) + person_locs[pidx + 1:]
                )
                new_state = _PlanState(elev_floors, new_locs)
                pid = self.person_ids[pidx]
                successors.append(
                    (f"EXIT{{{pid},{eid}}}", new_state, exit_cost)
                )

        return successors

    # ----------------------------------------------------------------- #
    # Heuristic (admissible & consistent under cost-shaping)            #
    # ----------------------------------------------------------------- #
    def _heuristic(self, plan_state):
        elev_floors = plan_state.elev_floors
        person_locs = plan_state.person_locs
        INF = 10 ** 9
        h = 0
        delivery_pairs = set()

        for pidx, loc in enumerate(person_locs):
            g = self.person_goal[pidx]
            if loc >= 0:
                if loc == g:
                    continue
                d = self.dist_to_goal[g].get(loc, INF)
                if d >= INF:
                    return INF
                h += 2 * d
            else:
                eidx = -loc - 1
                stints = self.min_stints_in_elev[eidx].get(g, INF)
                if stints >= INF:
                    return INF
                h += 2 * stints - 1
                if (g in self.elev_reachable[eidx]
                        and elev_floors[eidx] != g):
                    delivery_pairs.add((eidx, g))

        return h + len(delivery_pairs)

    # ----------------------------------------------------------------- #
    # Reward bookkeeping for A* tiebreak                                #
    # ----------------------------------------------------------------- #
    def _delivered_reward(self, plan_state):
        """Sum of mean rewards for persons currently at their goal floor.
        Used as a secondary heap key to break A* ties in favor of plans
        that deliver higher-reward persons earlier — helps when
        stochastic execution doesn't complete the full plan."""
        total = 0.0
        locs = plan_state.person_locs
        for pidx in range(len(locs)):
            if locs[pidx] == self.person_goal[pidx]:
                total += self.person_mean_reward[pidx]
        return total

    # ----------------------------------------------------------------- #
    # A* (cost-shaped, with reward-weighted tiebreak)                   #
    # ----------------------------------------------------------------- #
    def _a_star(self, start_state):
        """Returns a list of action strings, or None if unsolvable."""
        if self._is_goal_state(start_state):
            return []

        # priority queue of (f, -delivered_reward, counter, g, state, pid)
        # secondary key (-delivered) breaks ties toward states that have
        # already delivered higher cumulative reward.
        counter = 0
        h0 = self._heuristic(start_state)
        if h0 >= 10 ** 9:
            return None
        delivered0 = self._delivered_reward(start_state)
        open_heap = [(h0, -delivered0, counter, 0.0, start_state, -1)]
        parents = {0: (-1, None)}
        state_to_id = {start_state: 0}
        closed = {}
        g_score = {start_state: 0.0}

        while open_heap:
            f, _neg_d, _ctr, g, state, pid_marker = heapq.heappop(open_heap)

            if state in closed:
                continue
            closed[state] = True

            if self._is_goal_state(state):
                actions = []
                cur_id = state_to_id[state]
                while True:
                    pid_back, act = parents[cur_id]
                    if pid_back < 0:
                        break
                    actions.append(act)
                    cur_id = pid_back
                actions.reverse()
                return actions

            for action, next_state, cost in self._successors(state):
                new_g = g + cost
                old_g = g_score.get(next_state)
                if old_g is not None and old_g <= new_g:
                    continue
                g_score[next_state] = new_g
                h = self._heuristic(next_state)
                if h >= 10 ** 9:
                    continue
                counter += 1
                state_to_id[next_state] = counter
                parents[counter] = (state_to_id[state], action)
                delivered = self._delivered_reward(next_state)
                heapq.heappush(
                    open_heap,
                    (new_g + h, -delivered, counter, new_g,
                     next_state, counter),
                )

        return None  # unsolvable

    # ----------------------------------------------------------------- #
    # Plan utilities                                                    #
    # ----------------------------------------------------------------- #
    def _plan_cost_estimate(self, start_state, plan):
        """Estimate the cost-shaped total cost of executing `plan` from
        `start_state` (deterministic execution)."""
        total = 0.0
        cur = start_state
        for act in plan:
            for action, ns, cost in self._successors(cur):
                if action == act:
                    total += cost
                    cur = ns
                    break
        return total

    # ----------------------------------------------------------------- #
    # Simulate "success" of an action on an A2 state (for fast-path)    #
    # ----------------------------------------------------------------- #
    def _simulate_success(self, a2_state, action):
        elevators_t, persons_t, total_remaining = a2_state
        if action == "RESET":
            return self.game.get_initial_state()
        # Parse action
        try:
            kind, args = action.split("{")
            args = args.rstrip("}").split(",")
            a = int(args[0])
            b = int(args[1])
        except Exception:
            return None

        if kind == "MOVE":
            eid, target_f = a, b
            new_elevators = []
            for (e, fl, w) in elevators_t:
                if e == eid:
                    new_elevators.append((e, target_f, w))
                else:
                    new_elevators.append((e, fl, w))
            return (tuple(sorted(new_elevators)), persons_t, total_remaining)

        if kind == "ENTER":
            pid, eid = a, b
            # find person and elevator
            new_persons = []
            for (p, loc) in persons_t:
                if p == pid:
                    new_persons.append((p, ('in', eid)))
                else:
                    new_persons.append((p, loc))
            # update elevator weight
            person_w = self.person_weight[self.pidx_of[pid]]
            new_elevators = []
            for (e, fl, w) in elevators_t:
                if e == eid:
                    new_elevators.append((e, fl, w + person_w))
                else:
                    new_elevators.append((e, fl, w))
            return (tuple(sorted(new_elevators)),
                    tuple(sorted(new_persons)), total_remaining)

        if kind == "EXIT":
            pid, eid = a, b
            person_w = self.person_weight[self.pidx_of[pid]]
            # find elevator floor
            elev_floor = None
            for (e, fl, w) in elevators_t:
                if e == eid:
                    elev_floor = fl
                    break
            new_elevators = []
            for (e, fl, w) in elevators_t:
                if e == eid:
                    new_elevators.append((e, fl, w - person_w))
                else:
                    new_elevators.append((e, fl, w))
            # check if delivered (at goal)
            p_goal = self.person_goal[self.pidx_of[pid]]
            if elev_floor == p_goal:
                # delivered: removed from persons_t, total_remaining -=1
                new_persons = tuple(
                    sorted((p, loc) for (p, loc) in persons_t if p != pid)
                )
                new_remaining = total_remaining - 1
                if new_remaining == 0:
                    # episode completes, state resets to initial
                    return self.game.get_initial_state()
                return (tuple(sorted(new_elevators)),
                        new_persons, new_remaining)
            # not at goal: person on floor
            new_persons = []
            for (p, loc) in persons_t:
                if p == pid:
                    new_persons.append((p, ('floor', elev_floor)))
                else:
                    new_persons.append((p, loc))
            return (tuple(sorted(new_elevators)),
                    tuple(sorted(new_persons)), total_remaining)

        return None

    # ----------------------------------------------------------------- #
    # Legality check (defensive)                                        #
    # ----------------------------------------------------------------- #
    def _is_legal(self, a2_state, action):
        if action == "RESET":
            return True
        try:
            kind, args = action.split("{")
            args = args.rstrip("}").split(",")
            a = int(args[0])
            b = int(args[1])
        except Exception:
            return False

        elevators_t, persons_t, _ = a2_state
        elev_lookup = {e: (fl, w) for (e, fl, w) in elevators_t}
        person_lookup = {p: loc for (p, loc) in persons_t}

        if kind == "MOVE":
            eid, target_f = a, b
            if eid not in elev_lookup:
                return False
            if target_f not in self.reachable.get(eid, set()):
                return False
            return True
        if kind == "ENTER":
            pid, eid = a, b
            if eid not in elev_lookup:
                return False
            if pid not in person_lookup:
                return False
            loc = person_lookup[pid]
            if loc[0] != 'floor':
                return False
            elev_floor, cur_w = elev_lookup[eid]
            if loc[1] != elev_floor:
                return False
            w_p = self.person_weight[self.pidx_of[pid]]
            if cur_w + w_p > self.capacities[eid]:
                return False
            return True
        if kind == "EXIT":
            pid, eid = a, b
            if eid not in elev_lookup:
                return False
            if pid not in person_lookup:
                return False
            loc = person_lookup[pid]
            if loc[0] != 'in' or loc[1] != eid:
                return False
            return True
        return False

    # ----------------------------------------------------------------- #
    # V estimate for the RESET decision                                 #
    # ----------------------------------------------------------------- #
    def _v_estimate(self, a2_state, steps_remaining):
        """Rough analytical estimate of expected reward from `a2_state`
        given `steps_remaining` steps left.

        Idea:
          - Compute cost-shaped plan length from this state (≈ expected
            steps to finish the current episode).
          - Estimate reward from finishing current episode (probability
            of complete delivery * episode reward).
          - Estimate future episodes from initial state at episode rate.
        """
        if steps_remaining <= 0:
            return 0.0

        _, persons_t, total_remaining = a2_state
        if total_remaining == 0:
            # episode just completed: pretend we're at initial
            return (steps_remaining / self.expected_episode_steps
                    ) * self.value_per_episode

        # local plan from this state
        plan_state = self._a2_to_plan_state(a2_state)
        h = self._heuristic(plan_state)
        # rough lower bound on cost-shaped steps from h
        avg_action_cost = sum(self.elev_move_cost) / max(
            1, len(self.elev_move_cost)
        )
        est_steps = max(1.0, h * avg_action_cost)

        # local reward: undelivered persons' mean rewards + goal_reward
        local_undelivered_reward = sum(
            self.person_mean_reward[self.pidx_of[pid]]
            for pid, _ in persons_t
        )
        local_value = local_undelivered_reward + self.goal_reward

        if est_steps >= steps_remaining:
            # can't finish: scale by progress
            return local_value * (steps_remaining / est_steps)

        local_part = local_value
        future_steps = steps_remaining - est_steps
        future_episodes = future_steps / self.expected_episode_steps
        return local_part + future_episodes * self.value_per_episode

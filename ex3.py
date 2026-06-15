"""
AI disclosure:
  Used: Claude Code for brainstorming the controller's design and writing
  the implementation. I (Shaked) directed the design through discussion,
  verified correctness against the specification, ran the local checker,
  and validated the implementation.
"""

import ext_elev
import heapq
import time
import numpy as np
from collections import deque

id = ["208904839"]


class _PlanState:
    """Deterministic planning state for the A* fallback.
    elev_floors: tuple[int]  -- floor of each elevator (sorted-ID order)
    person_locs: tuple[int]  -- per person:
        loc >= 0 -> standing on floor `loc`
        loc <  0 -> inside elevator with idx (-loc - 1)
    """
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
    """Reinforcement-learning multi-elevator controller.

    Two-tier strategy:

      Tier 1 -- Stationary discounted value iteration (gamma=0.98) on the
      reachable MDP under the posterior-mean estimate of the hidden model.
      Optimal for the estimated model; per-step cost is an O(1) policy
      lookup. Used whenever the BFS-reachable state set fits the cap.

      Tier 2 -- Cost-shaped A* fallback (ported from Assignment 2) with
      subset-strategy selection for huge problems. The A* heuristic stays
      admissible-consistent under cost-shaping; the action costs are
      derived from the same posterior-mean probabilities.

    Bayesian model:
      * Beta(2, 1) prior on each elevator MOVE / person ENTER+EXIT
        success probability, updated from observed outcomes.
      * Per-person reward samples accumulated from observed deliveries
        (with goal_reward subtracted when the global goal triggers).

    Replanning is gated by both a step-count schedule (gates at 25, 100)
    and by surprise detection (any active entity probability shift >=0.10);
    a hard time-budget gate (55% of the seed budget) stops replanning so
    we never risk a per-seed timeout (TIMEOUT -> 0 reward).
    """

    _MDP_DELIVERED = 1_000_000
    _GAMMA = 0.98
    _VI_TOL = 1e-3
    _VI_MAX_ITERS = 200
    _BFS_STATE_CAP = 120_000
    _BFS_TIME_FRAC = 0.10          # bound BFS to this fraction of budget
    _FIRST_PLAN_TIME_FRAC = 0.22
    _REPLAN_TIME_FRAC = 0.10
    _TIME_BUDGET_GATE = 0.55       # stop replanning once we burned this much
    _HARD_STOP_FRAC = 0.92         # past this, return a safe action immediately
    _SHIFT_TRIGGER = 0.10          # |Delta p| trigger for surprise replan
    _PRIOR_ALPHA = 2.0             # Beta(2, 1) -> mean 0.667, modest optimism
    _PRIOR_BETA = 1.0
    _REWARD_PRIOR = 5.0            # placeholder mean until first delivery
    _REWARD_SAMPLES_CAP = 50
    _MIN_REPLAN_INTERVAL = 10      # min steps between consecutive replans

    # ----------------------------------------------------------------- #
    # Construction                                                       #
    # ----------------------------------------------------------------- #
    def __init__(self, game: ext_elev.GameAPI):
        self.game = game
        self._start_time = time.perf_counter()

        # ---- static info via public getters --------------------------- #
        self.reachable = game.get_reachable()
        self.capacities = game.get_capacities()
        initial_state = game.get_initial_state()
        _, persons_t0, _ = initial_state

        self.elevator_ids = sorted(self.reachable.keys())
        self.person_ids = sorted(pid for pid, _ in persons_t0)
        self.eidx_of = {e: i for i, e in enumerate(self.elevator_ids)}
        self.pidx_of = {p: i for i, p in enumerate(self.person_ids)}

        self.elev_reachable = [self.reachable[e] for e in self.elevator_ids]
        self.elev_capacity = [self.capacities[e] for e in self.elevator_ids]
        self.person_weight = [
            game.get_person_weight(p) for p in self.person_ids
        ]
        self.person_goal = [
            game.get_person_goal(p) for p in self.person_ids
        ]
        self.goal_locs_tuple = tuple(self.person_goal)

        self.goal_reward = float(game.get_goal_reward())
        self.max_steps = int(game.get_max_steps())
        self.budget = 20.0 + 0.5 * self.max_steps

        self._initial_state = initial_state
        self._mdp_init_state = self._mdp_encode(initial_state)

        # ---- posteriors on the hidden model -------------------------- #
        ne = len(self.elevator_ids)
        npe = len(self.person_ids)
        self._post_e_a = [self._PRIOR_ALPHA] * ne
        self._post_e_b = [self._PRIOR_BETA] * ne
        self._post_p_a = [self._PRIOR_ALPHA] * npe
        self._post_p_b = [self._PRIOR_BETA] * npe
        self._reward_samples = [[] for _ in range(npe)]
        self._refresh_estimates()

        # Tier-1 MDP planning state
        self._policy = None
        self._last_plan_e = None
        self._last_plan_p = None
        self._last_plan_r = None        # snapshot of mean rewards at last plan
        self._next_replan_step = 15
        self._replan_gates = [15, 40, 100]
        self._mdp_too_big = False
        # Cached reachable-state list + structure tensors. Topology never
        # changes within a seed (probabilities change, the graph doesn't),
        # so BFS + structure build run once; replans re-use them.
        self._cached_states = None
        self._cached_state_to_idx = None
        self._cached_sa_action = None
        self._cached_sa_outcome_off = None
        self._cached_action_off = None
        self._cached_out_next = None
        self._cached_V = None              # warm-start across replans
        # Force-replan flag: set whenever an observation gives us a delivery
        # reward sample we haven't planned with yet. Bypasses the step gate
        # — critical for reset-loop detection on rl_*.
        self._force_replan = False
        self._last_replan_step = -1000     # step number of last replan

        # Tier-2 A* fallback state
        self._astar_ready = False
        self._astar_plan = None
        self._astar_plan_idx = 0
        self._astar_expected_state = None
        self._astar_target_subset = None

        self._last_obs_state = None
        self._last_obs_action = None

        # ---- precomputations for heuristics & A* --------------------- #
        self._compute_floor_adj()
        self._compute_transitive_reach()
        self._compute_useful_exit_floors()
        self._compute_min_stints_in_elev()
        self._compute_elev_overlap()
        self._compute_transfer_floors()

        # ---- first plan (uses optimistic prior model) ---------------- #
        first_deadline = (
            self._start_time + self._FIRST_PLAN_TIME_FRAC * self.budget
        )
        self._plan_now(first_deadline)

    # ----------------------------------------------------------------- #
    # Public API                                                         #
    # ----------------------------------------------------------------- #
    def choose_next_action(self, state):
        # 1. Learn from the previous step's outcome.
        if self._last_obs_state is not None:
            self._observe(self._last_obs_state, self._last_obs_action, state)

        time_used_frac = (time.perf_counter() - self._start_time) / self.budget
        # Hard-stop guard
        if time_used_frac > self._HARD_STOP_FRAC:
            action = self._quick_safe_action(state)
            self._last_obs_state = state
            self._last_obs_action = action
            return action

        # 2. Maybe replan.
        cur_step = self.game.get_current_steps()
        if (time_used_frac < self._TIME_BUDGET_GATE
                and self._should_replan(cur_step)):
            time_left = self.budget - (time.perf_counter() - self._start_time)
            replan_budget = min(
                self._REPLAN_TIME_FRAC * self.budget, 0.35 * time_left
            )
            if replan_budget > 4.0:
                self._plan_now(time.perf_counter() + replan_budget)
                self._last_replan_step = cur_step
            self._advance_replan_gate(cur_step)

        # 3. Choose action: policy lookup if available, else A* plan
        action = self._choose_action(state)

        self._last_obs_state = state
        self._last_obs_action = action
        return action

    # ----------------------------------------------------------------- #
    # Posterior bookkeeping                                              #
    # ----------------------------------------------------------------- #
    def _observe(self, prev_state, prev_action, new_state):
        if prev_action == "RESET":
            return
        kind, parts = self._parse_action(prev_action)
        if kind is None:
            return
        last_reward = float(self.game.get_last_gained_reward())

        elev_prev = {e: (fl, w) for e, fl, w in prev_state[0]}
        elev_new = {e: (fl, w) for e, fl, w in new_state[0]}
        pers_prev = {p: loc for p, loc in prev_state[1]}
        pers_new = {p: loc for p, loc in new_state[1]}

        if kind == "MOVE":
            eid, target = parts
            if eid not in self.eidx_of:
                return
            eidx = self.eidx_of[eid]
            new_fl = elev_new.get(eid, (None,))[0]
            if new_fl == target:
                self._post_e_a[eidx] += 1.0
            else:
                self._post_e_b[eidx] += 1.0

        elif kind == "ENTER":
            pid, eid = parts
            if pid not in self.pidx_of:
                return
            pidx = self.pidx_of[pid]
            old_loc = pers_prev.get(pid)
            new_loc = pers_new.get(pid)
            if (old_loc is not None and new_loc is not None
                    and old_loc[0] == 'floor' and new_loc[0] == 'in'):
                self._post_p_a[pidx] += 1.0
            else:
                self._post_p_b[pidx] += 1.0

        elif kind == "EXIT":
            pid, eid = parts
            if pid not in self.pidx_of:
                return
            pidx = self.pidx_of[pid]
            delivered = (
                pid not in pers_new or new_state == self._initial_state
            )
            if delivered:
                self._post_p_a[pidx] += 1.0
                if last_reward > 0:
                    sample = (
                        last_reward - self.goal_reward
                        if new_state == self._initial_state
                        else last_reward
                    )
                    if sample > 0:
                        was_first = (len(self._reward_samples[pidx]) == 0)
                        self._reward_samples[pidx].append(float(sample))
                        if len(self._reward_samples[pidx]) > self._REWARD_SAMPLES_CAP:
                            self._reward_samples[pidx] = self._reward_samples[
                                pidx
                            ][-self._REWARD_SAMPLES_CAP:]
                        # First-ever sample for this person flips us out of
                        # the prior — force a replan immediately so the MDP
                        # re-evaluates using the observed reward magnitude.
                        if was_first:
                            self._force_replan = True
            else:
                old_loc = pers_prev.get(pid)
                new_loc = pers_new.get(pid)
                if (old_loc is not None and new_loc is not None
                        and old_loc[0] == 'in' and new_loc[0] == 'floor'):
                    self._post_p_a[pidx] += 1.0
                else:
                    self._post_p_b[pidx] += 1.0

        self._refresh_estimates()

    def _refresh_estimates(self):
        self._elev_p = [
            a / (a + b) for a, b in zip(self._post_e_a, self._post_e_b)
        ]
        self._person_p = [
            a / (a + b) for a, b in zip(self._post_p_a, self._post_p_b)
        ]
        self._person_mean_r = [
            (sum(s) / len(s)) if s else self._REWARD_PRIOR
            for s in self._reward_samples
        ]
        # Cost-shaped action costs (used by the A* fallback). Soften the
        # 1/p admissibility-tight cost with alpha=0.5: failed MOVEs still
        # let us interleave useful work elsewhere, so over-charging the
        # broken elevators ruins the heuristic. The same softening was
        # used in Assignment 2.
        self._elev_move_cost = [
            1.0 + 0.5 * (1.0 / max(p, 0.05) - 1.0) for p in self._elev_p
        ]
        self._person_action_cost = [
            1.0 + 0.5 * (1.0 / max(p, 0.05) - 1.0) for p in self._person_p
        ]

    def _should_replan(self, cur_step):
        if self._mdp_too_big:
            return False
        if self._policy is None:
            return True
        # Min interval throttle (bypassed by force-replan on first delivery).
        if not self._force_replan:
            if cur_step - self._last_replan_step < self._MIN_REPLAN_INTERVAL:
                return False
        if self._force_replan:
            return True
        # Otherwise, refine at the scheduled gates only. Early prob-shift
        # bypass turned out to be noisy — refines mid-flight on a half-
        # observed posterior, which hurt p1_hard / e2_hard. The scheduled
        # gates (with a few extra early ones) give us better stability.
        return cur_step >= self._next_replan_step

    def _advance_replan_gate(self, cur_step):
        future_gates = [g for g in self._replan_gates if g > cur_step]
        if future_gates:
            self._next_replan_step = min(future_gates)
        else:
            self._next_replan_step = cur_step + 100

    # ----------------------------------------------------------------- #
    # Planning -- top-level dispatcher                                   #
    # ----------------------------------------------------------------- #
    def _plan_now(self, deadline):
        """Attempt Tier-1 (MDP+VI). On BFS cap, switch to Tier-2 (A*)
        and pick the best target subset. Once Tier-2 is committed, we
        never come back to Tier-1 (would just waste BFS time).

        On the first call we enumerate reachable states and build the
        topology-only structure (sa_action, sa_outcome_off, action_off,
        out_next — all probability-independent). On replans we re-use
        the cache: only sa_imm and out_prob change."""
        if self._cached_states is None:
            bfs_deadline = min(
                deadline,
                time.perf_counter() + self._BFS_TIME_FRAC * self.budget,
            )
            seen = {self._mdp_init_state}
            frontier = deque([self._mdp_init_state])
            cap_hit = False
            while frontier:
                if time.perf_counter() > bfs_deadline:
                    cap_hit = True
                    break
                if len(seen) > self._BFS_STATE_CAP:
                    cap_hit = True
                    break
                s = frontier.popleft()
                for a in self._mdp_legal_actions(s):
                    for _p, ns, _r in self._mdp_outcomes(s, a):
                        if ns not in seen:
                            seen.add(ns)
                            frontier.append(ns)
            if cap_hit:
                self._mdp_too_big = True
                self._policy = None
                self._setup_astar_tier()
                return
            if time.perf_counter() > deadline:
                return
            self._build_topology_cache(list(seen), deadline)
            if self._cached_states is None:
                return  # deadline hit before structure built

        self._refresh_transitions_and_solve(deadline)

    # ----------------------------------------------------------------- #
    # Tier 1 -- MDP build + stationary discounted VI                     #
    # ----------------------------------------------------------------- #
    # Outcome type codes (kept as small int constants for fast indexing).
    _OT_RESET = 0
    _OT_MOVE_S = 1
    _OT_MOVE_F = 2
    _OT_ENTER_S = 3
    _OT_ENTER_F = 4
    _OT_EXIT_S = 5
    _OT_EXIT_F = 6

    def _build_topology_cache(self, states, deadline):
        """One-shot build of the probability-independent MDP structure.

        Outcome data is stored as parallel numpy arrays so that the
        per-replan refresh is fully vectorised (no Python loop over the
        millions of outcomes for the larger m_* problems):

          out_type[i]       : outcome category (int8, 0..6)
          out_entity[i]     : eidx for MOVE_*, pidx for ENTER_* / EXIT_*
          out_n_others[i]   : |F_e ∪ {f0} \\ {target}| for MOVE_F (else 1)
          out_is_goal[i]    : EXIT_S at the person's goal floor
          out_all_done[i]   : EXIT_S that empties the persons list (auto-reset)
        """
        n_states = len(states)
        state_to_idx = {s: i for i, s in enumerate(states)}

        sa_action = []
        sa_outcome_off = [0]
        action_off = [0]
        out_next = []
        out_type = []
        out_entity = []
        out_n_others = []
        out_is_goal = []
        out_all_done = []
        sa_total = 0
        out_total = 0

        for s in states:
            for a in self._mdp_legal_actions(s):
                outs = self._mdp_outcomes(s, a)
                fields = self._mdp_outcome_fields(s, a)
                for (prob, ns, r), f in zip(outs, fields):
                    out_next.append(state_to_idx[ns])
                    out_type.append(f[0])
                    out_entity.append(f[1])
                    out_n_others.append(f[2])
                    out_is_goal.append(f[3])
                    out_all_done.append(f[4])
                    out_total += 1
                sa_action.append(a)
                sa_outcome_off.append(out_total)
                sa_total += 1
            action_off.append(sa_total)
            if time.perf_counter() > deadline:
                return

        self._cached_states = states
        self._cached_state_to_idx = state_to_idx
        self._cached_sa_action = sa_action
        self._cached_sa_outcome_off = np.asarray(
            sa_outcome_off, dtype=np.int64
        )
        self._cached_action_off = np.asarray(action_off, dtype=np.int64)
        self._cached_out_next = np.asarray(out_next, dtype=np.int32)
        self._cached_out_type = np.asarray(out_type, dtype=np.int8)
        self._cached_out_entity = np.asarray(out_entity, dtype=np.int32)
        self._cached_out_n_others = np.asarray(
            out_n_others, dtype=np.float64
        )
        self._cached_out_is_goal = np.asarray(out_is_goal, dtype=bool)
        self._cached_out_all_done = np.asarray(out_all_done, dtype=bool)
        # Pre-compute masks once.
        self._cached_mask_reset = self._cached_out_type == self._OT_RESET
        self._cached_mask_move_s = self._cached_out_type == self._OT_MOVE_S
        self._cached_mask_move_f = self._cached_out_type == self._OT_MOVE_F
        self._cached_mask_enter_s = self._cached_out_type == self._OT_ENTER_S
        self._cached_mask_enter_f = self._cached_out_type == self._OT_ENTER_F
        self._cached_mask_exit_s = self._cached_out_type == self._OT_EXIT_S
        self._cached_mask_exit_f = self._cached_out_type == self._OT_EXIT_F
        self._cached_V = self._heuristic_v0(states)

    def _mdp_outcome_fields(self, state, action):
        """Return per-outcome (type, entity, n_others, is_goal, all_done)
        tuples aligned with _mdp_outcomes(state, action)."""
        ef, pl = state
        DELIV = self._MDP_DELIVERED
        kind = action[0]
        if kind == 'RESET':
            return [(self._OT_RESET, 0, 1, False, False)]
        if kind == 'MOVE':
            _, eidx, target = action
            others_set = set(self.elev_reachable[eidx]) - {target}
            others_set.add(ef[eidx])
            n_others = max(1, len(others_set))
            out = [(self._OT_MOVE_S, eidx, 1, False, False)]
            for _ in range(len(others_set)):
                out.append((self._OT_MOVE_F, eidx, n_others, False, False))
            return out
        if kind == 'ENTER':
            _, pidx, eidx = action
            return [
                (self._OT_ENTER_S, pidx, 1, False, False),
                (self._OT_ENTER_F, pidx, 1, False, False),
            ]
        # EXIT
        _, pidx, eidx = action
        floor = ef[eidx]
        if floor == self.person_goal[pidx]:
            pl_s = list(pl); pl_s[pidx] = DELIV
            all_done = all(x == DELIV for x in pl_s)
            return [
                (self._OT_EXIT_S, pidx, 1, True, all_done),
                (self._OT_EXIT_F, pidx, 1, False, False),
            ]
        return [
            (self._OT_EXIT_S, pidx, 1, False, False),
            (self._OT_EXIT_F, pidx, 1, False, False),
        ]

    def _refresh_transitions_and_solve(self, deadline):
        """Vectorised refresh of out_prob + sa_imm from current estimates,
        then warm-started discounted VI."""
        n_out = self._cached_out_type.shape[0]
        elev_p = np.asarray(self._elev_p, dtype=np.float64)
        person_p = np.asarray(self._person_p, dtype=np.float64)
        person_r = np.asarray(self._person_mean_r, dtype=np.float64)
        entity = self._cached_out_entity

        out_prob = np.empty(n_out, dtype=np.float64)
        # RESET → deterministic
        out_prob[self._cached_mask_reset] = 1.0
        # MOVE success / failure
        m = self._cached_mask_move_s
        out_prob[m] = elev_p[entity[m]]
        m = self._cached_mask_move_f
        out_prob[m] = (1.0 - elev_p[entity[m]]) / self._cached_out_n_others[m]
        # ENTER success / failure
        m = self._cached_mask_enter_s
        out_prob[m] = person_p[entity[m]]
        m = self._cached_mask_enter_f
        out_prob[m] = 1.0 - person_p[entity[m]]
        # EXIT success / failure
        m = self._cached_mask_exit_s
        out_prob[m] = person_p[entity[m]]
        m = self._cached_mask_exit_f
        out_prob[m] = 1.0 - person_p[entity[m]]

        # Per-outcome reward: only EXIT_S at goal contributes.
        out_reward = np.zeros(n_out, dtype=np.float64)
        m_goal = self._cached_mask_exit_s & self._cached_out_is_goal
        out_reward[m_goal] = person_r[entity[m_goal]]
        m_full = m_goal & self._cached_out_all_done
        out_reward[m_full] += self.goal_reward

        sa_outcome_off = self._cached_sa_outcome_off
        sa_starts = sa_outcome_off[:-1]
        sa_imm = np.add.reduceat(out_prob * out_reward, sa_starts)

        if time.perf_counter() > deadline:
            return

        out_next = self._cached_out_next
        action_off = self._cached_action_off
        action_starts = action_off[:-1]

        V = self._cached_V.copy()
        gamma = self._GAMMA
        for _ in range(self._VI_MAX_ITERS):
            if time.perf_counter() > deadline:
                break
            contrib = out_prob * V[out_next]
            sa_future = np.add.reduceat(contrib, sa_starts)
            Q = sa_imm + gamma * sa_future
            V_new = np.maximum.reduceat(Q, action_starts)
            if np.abs(V_new - V).max() < self._VI_TOL:
                V = V_new
                break
            V = V_new

        contrib = out_prob * V[out_next]
        sa_future = np.add.reduceat(contrib, sa_starts)
        Q = sa_imm + gamma * sa_future

        n_states = len(self._cached_states)
        sa_action = self._cached_sa_action
        states = self._cached_states
        policy = {}
        for s_idx in range(n_states):
            start = int(action_off[s_idx])
            end = int(action_off[s_idx + 1])
            if start == end:
                continue
            local = int(Q[start:end].argmax())
            policy[states[s_idx]] = sa_action[start + local]

        self._policy = policy
        self._cached_V = V  # warm-start next replan
        self._last_plan_e = list(self._elev_p)
        self._last_plan_p = list(self._person_p)
        self._last_plan_r = list(self._person_mean_r)
        self._force_replan = False

    def _heuristic_v0(self, states):
        arr = np.empty(len(states), dtype=np.float64)
        for i, s in enumerate(states):
            arr[i] = self._h_value(s)
        return arr

    def _h_value(self, state):
        """Cost-shaped expected-reward heuristic. Used to seed V0 for VI."""
        ef, pl = state
        DELIV = self._MDP_DELIVERED
        score = 0.0
        max_dist = 12
        n_remaining = 0
        for pidx, loc in enumerate(pl):
            mean_r = self._person_mean_r[pidx]
            goal = self.person_goal[pidx]
            if loc == DELIV:
                score += mean_r
                continue
            n_remaining += 1
            if loc < 0:
                eidx = -loc - 1
                f = ef[eidx]
                if f == goal:
                    score += mean_r * 0.85
                else:
                    d = self._dist_to_goal.get(goal, {}).get(f, max_dist)
                    score += mean_r * max(0.0, 0.55 - 0.05 * d)
            elif loc == goal:
                score += mean_r * 0.45
            else:
                d = self._dist_to_goal.get(goal, {}).get(loc, max_dist)
                score += mean_r * max(0.0, 0.20 - 0.02 * d)
        if n_remaining == 0:
            score += self.goal_reward

        remaining_horizon = max(
            0, self.max_steps - self.game.get_current_steps()
        )
        if remaining_horizon > 20:
            episode_R = sum(self._person_mean_r) + self.goal_reward
            cycle_len = max(15.0, 4.0 * len(self.person_ids))
            n_cycles = remaining_horizon / cycle_len
            score += episode_R * n_cycles * 0.25
        return score

    # ----------------------------------------------------------------- #
    # Action selection                                                   #
    # ----------------------------------------------------------------- #
    def _choose_action(self, state):
        # Tier 1: MDP policy lookup
        if self._policy is not None:
            s = self._mdp_encode(state)
            a = self._policy.get(s)
            if a is not None:
                return self._mdp_action_str(a)
        # Tier 2: A* plan execution
        if self._astar_ready:
            return self._astar_choose(state)
        # Fallback fallback: 1-step lookahead under estimated model
        return self._one_step_lookahead(self._mdp_encode(state))

    def _one_step_lookahead(self, encoded_state):
        acts = self._mdp_legal_actions(encoded_state)
        if not acts:
            return "RESET"
        best_q = -float('inf')
        best_a = ('RESET',)
        gamma = self._GAMMA
        for a in acts:
            q = 0.0
            for prob, ns, r in self._mdp_outcomes(encoded_state, a):
                q += prob * (r + gamma * self._h_value(ns))
            if q > best_q:
                best_q = q
                best_a = a
        return self._mdp_action_str(best_a)

    def _quick_safe_action(self, state):
        """Last-resort cheap action when the time budget is almost gone."""
        s = self._mdp_encode(state)
        acts = self._mdp_legal_actions(s)
        for a in acts:
            if a[0] == 'EXIT':
                return self._mdp_action_str(a)
        for a in acts:
            if a[0] == 'ENTER':
                return self._mdp_action_str(a)
        for a in acts:
            if a[0] == 'MOVE':
                return self._mdp_action_str(a)
        return "RESET"

    # ----------------------------------------------------------------- #
    # Tier 2 -- Cost-shaped A* fallback (subset strategy + replan)       #
    # ----------------------------------------------------------------- #
    def _setup_astar_tier(self):
        """Pick the best target subset and seed an initial plan."""
        self._pick_best_strategy()
        self._astar_plan = self._a_star_for_subset(
            self._initial_state, self._astar_target_subset
        )
        self._astar_plan_idx = 0
        if self._astar_plan:
            self._astar_expected_state = self._simulate_success(
                self._initial_state, self._astar_plan[0]
            )
        else:
            self._astar_expected_state = None
        self._astar_ready = True

    def _astar_choose(self, state):
        _, persons_t, _ = state
        undelivered_ids = {pid for pid, _ in persons_t}

        # If our target subset is fully delivered, RESET to fetch again.
        if not (self._astar_target_subset & undelivered_ids):
            self._astar_plan = None
            self._astar_plan_idx = 0
            self._astar_expected_state = self._initial_state
            return "RESET"

        # Fast path: state matches expected -> advance plan
        if (self._astar_expected_state is not None
                and state == self._astar_expected_state
                and self._astar_plan
                and self._astar_plan_idx < len(self._astar_plan)):
            action = self._astar_plan[self._astar_plan_idx]
            self._astar_plan_idx += 1
        else:
            self._astar_plan = self._a_star_for_subset(
                state, self._astar_target_subset
            )
            self._astar_plan_idx = 0
            if self._astar_plan:
                action = self._astar_plan[0]
                self._astar_plan_idx = 1
            else:
                self._astar_plan = None
                self._astar_plan_idx = 0
                self._astar_expected_state = self._initial_state
                return "RESET"

        if not self._is_legal(state, action):
            self._astar_plan = None
            self._astar_plan_idx = 0
            self._astar_expected_state = self._initial_state
            return "RESET"

        self._astar_expected_state = self._simulate_success(state, action)
        return action

    def _pick_best_strategy(self):
        """Pick the target subset with highest reward/step ratio,
        considering reset-loop strategies (RESET after partial delivery)
        and full-delivery (auto-reset via goal_reward)."""
        import itertools as _it
        n = len(self.person_ids)
        candidates = [frozenset(self.person_ids)]
        max_k = 3 if n <= 5 else 2
        for k in range(1, min(max_k, n) + 1):
            for combo in _it.combinations(self.person_ids, k):
                candidates.append(frozenset(combo))

        best_per_step = -1.0
        best_subset = frozenset(self.person_ids)

        for subset in candidates:
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
                reward = sum(self._person_mean_r) + self.goal_reward
                eff_cost = cost
            else:
                reward = sum(
                    self._person_mean_r[self.pidx_of[pid]] for pid in subset
                )
                eff_cost = cost + 1.0
            per_step = reward / eff_cost
            if per_step > best_per_step:
                best_per_step = per_step
                best_subset = subset

        self._astar_target_subset = best_subset

    # ----------------------------------------------------------------- #
    # A2 precomputations (BFS / connectivity)                            #
    # ----------------------------------------------------------------- #
    def _compute_floor_adj(self):
        floors = set()
        for r in self.elev_reachable:
            floors |= set(r)
        floors.update(self.person_goal)

        adj = {f: set() for f in floors}
        for r in self.elev_reachable:
            lst = list(r)
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    adj[lst[i]].add(lst[j])
                    adj[lst[j]].add(lst[i])

        goal_floors = set(self.person_goal)
        self._dist_to_goal = {}
        for g in goal_floors:
            d = {g: 0}
            q = deque([g])
            while q:
                u = q.popleft()
                for v in adj.get(u, ()):
                    if v not in d:
                        d[v] = d[u] + 1
                        q.append(v)
            self._dist_to_goal[g] = d

    def _compute_transitive_reach(self):
        n = len(self.elevator_ids)
        e_adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if self.elev_reachable[i] & self.elev_reachable[j]:
                    e_adj[i].add(j)
                    e_adj[j].add(i)
        self._elev_transitive = []
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
            self._elev_transitive.append(frozenset(union))

    def _compute_useful_exit_floors(self):
        self._useful_exit = []
        for pidx in range(len(self.person_ids)):
            g = self.person_goal[pidx]
            useful = {g}
            for eidx2 in range(len(self.elevator_ids)):
                if g in self._elev_transitive[eidx2]:
                    useful |= self.elev_reachable[eidx2]
            self._useful_exit.append(frozenset(useful))

    def _compute_min_stints_in_elev(self):
        n = len(self.elevator_ids)
        INF = 10 ** 9
        self._min_stints_in_elev = [dict() for _ in range(n)]
        goal_set = set(self.person_goal)
        for eidx in range(n):
            reach = self.elev_reachable[eidx]
            for g in goal_set:
                if g in reach:
                    self._min_stints_in_elev[eidx][g] = 1
                else:
                    g_dist = self._dist_to_goal[g]
                    best = INF
                    for fp in reach:
                        d = g_dist.get(fp, INF)
                        if d < best:
                            best = d
                    self._min_stints_in_elev[eidx][g] = (
                        1 + best if best < INF else INF
                    )

    def _compute_elev_overlap(self):
        n = len(self.elevator_ids)
        self._elev_overlap = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                self._elev_overlap[(i, j)] = (
                    self.elev_reachable[i] & self.elev_reachable[j]
                )

    def _compute_transfer_floors(self):
        n = len(self.elevator_ids)
        self._transfer_floors = [dict() for _ in range(n)]
        goal_set = set(self.person_goal)
        for eidx in range(n):
            for g in goal_set:
                if g in self.elev_reachable[eidx]:
                    continue
                acc = set()
                for other in range(n):
                    if other == eidx:
                        continue
                    if g in self._elev_transitive[other]:
                        acc |= self._elev_overlap[(eidx, other)]
                self._transfer_floors[eidx][g] = frozenset(acc)

    # ----------------------------------------------------------------- #
    # A* search (cost-shaped, admissible heuristic)                      #
    # ----------------------------------------------------------------- #
    def _a_star_for_subset(self, a2_state, subset):
        plan_state = self._a2_to_subset_plan_state(a2_state, subset)
        return self._a_star(plan_state)

    def _a2_to_subset_plan_state(self, a2_state, subset):
        elevators_t, persons_t, _ = a2_state
        elev_floors = [0] * len(self.elevator_ids)
        for eid, fl, _w in elevators_t:
            elev_floors[self.eidx_of[eid]] = fl
        person_locs = list(self.person_goal)
        persons_lookup = {pid: loc for pid, loc in persons_t}
        for pid in subset:
            if pid not in persons_lookup:
                continue
            loc = persons_lookup[pid]
            idx = self.pidx_of[pid]
            if loc[0] == 'floor':
                person_locs[idx] = loc[1]
            else:
                person_locs[idx] = -self.eidx_of[loc[1]] - 1
        return _PlanState(tuple(elev_floors), tuple(person_locs))

    def _is_goal_state(self, plan_state):
        return plan_state.person_locs == self.goal_locs_tuple

    def _successors(self, plan_state):
        elev_floors = plan_state.elev_floors
        person_locs = plan_state.person_locs
        n_elev = len(self.elevator_ids)
        n_pers = len(self.person_ids)

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

        elev_must_exit = [False] * n_elev
        for eidx in range(n_elev):
            ef = elev_floors[eidx]
            for _, g in in_elev_persons[eidx]:
                if g == ef:
                    elev_must_exit[eidx] = True
                    break

        successors = []

        # MOVE
        for eidx in range(n_elev):
            if elev_must_exit[eidx]:
                continue
            cur_floor = elev_floors[eidx]
            reach = self.elev_reachable[eidx]
            elev_trans = self._elev_transitive[eidx]
            move_cost = self._elev_move_cost[eidx]
            candidates = set()
            for _, loc, g in on_floor_persons:
                if loc != g and loc in reach and g in elev_trans:
                    candidates.add(loc)
            tf_for_e = self._transfer_floors[eidx]
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

        # ENTER
        for pidx, loc, g in on_floor_persons:
            if loc == g:
                continue
            pid = self.person_ids[pidx]
            w = self.person_weight[pidx]
            pre = person_locs[:pidx]
            post = person_locs[pidx + 1:]
            enter_cost = self._person_action_cost[pidx]
            for eidx in range(n_elev):
                if elev_floors[eidx] != loc:
                    continue
                if elev_must_exit[eidx]:
                    continue
                if g not in self._elev_transitive[eidx]:
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

        # EXIT
        for eidx in range(n_elev):
            ef = elev_floors[eidx]
            eid = self.elevator_ids[eidx]
            for pidx, g in in_elev_persons[eidx]:
                if ef not in self._useful_exit[pidx]:
                    continue
                exit_cost = self._person_action_cost[pidx]
                new_locs = (
                    person_locs[:pidx] + (ef,) + person_locs[pidx + 1:]
                )
                new_state = _PlanState(elev_floors, new_locs)
                pid = self.person_ids[pidx]
                successors.append(
                    (f"EXIT{{{pid},{eid}}}", new_state, exit_cost)
                )

        return successors

    def _a_star_heuristic(self, plan_state):
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
                d = self._dist_to_goal[g].get(loc, INF)
                if d >= INF:
                    return INF
                h += 2 * d
            else:
                eidx = -loc - 1
                stints = self._min_stints_in_elev[eidx].get(g, INF)
                if stints >= INF:
                    return INF
                h += 2 * stints - 1
                if (g in self.elev_reachable[eidx]
                        and elev_floors[eidx] != g):
                    delivery_pairs.add((eidx, g))
        return h + len(delivery_pairs)

    def _delivered_reward(self, plan_state):
        total = 0.0
        locs = plan_state.person_locs
        for pidx in range(len(locs)):
            if locs[pidx] == self.person_goal[pidx]:
                total += self._person_mean_r[pidx]
        return total

    def _a_star(self, start_state):
        if self._is_goal_state(start_state):
            return []
        counter = 0
        h0 = self._a_star_heuristic(start_state)
        if h0 >= 10 ** 9:
            return None
        delivered0 = self._delivered_reward(start_state)
        open_heap = [(h0, -delivered0, counter, 0.0, start_state, -1)]
        parents = {0: (-1, None)}
        state_to_id = {start_state: 0}
        closed = {}
        g_score = {start_state: 0.0}

        while open_heap:
            f, _neg_d, _ctr, g, state, _pid = heapq.heappop(open_heap)
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
                h = self._a_star_heuristic(next_state)
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
        return None

    def _plan_cost_estimate(self, start_state, plan):
        total = 0.0
        cur = start_state
        for act in plan:
            for action, ns, cost in self._successors(cur):
                if action == act:
                    total += cost
                    cur = ns
                    break
        return total

    def _simulate_success(self, a2_state, action):
        elevators_t, persons_t, total_remaining = a2_state
        if action == "RESET":
            return self._initial_state
        try:
            kind, args = action.split("{")
            args = args.rstrip("}").split(",")
            a = int(args[0]); b = int(args[1])
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
            new_persons = []
            for (p, loc) in persons_t:
                if p == pid:
                    new_persons.append((p, ('in', eid)))
                else:
                    new_persons.append((p, loc))
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
            p_goal = self.person_goal[self.pidx_of[pid]]
            if elev_floor == p_goal:
                new_persons = tuple(
                    sorted((p, loc) for (p, loc) in persons_t if p != pid)
                )
                new_remaining = total_remaining - 1
                if new_remaining == 0:
                    return self._initial_state
                return (tuple(sorted(new_elevators)),
                        new_persons, new_remaining)
            new_persons = []
            for (p, loc) in persons_t:
                if p == pid:
                    new_persons.append((p, ('floor', elev_floor)))
                else:
                    new_persons.append((p, loc))
            return (tuple(sorted(new_elevators)),
                    tuple(sorted(new_persons)), total_remaining)
        return None

    def _is_legal(self, a2_state, action):
        if action == "RESET":
            return True
        try:
            kind, args = action.split("{")
            args = args.rstrip("}").split(",")
            a = int(args[0]); b = int(args[1])
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
    # MDP machinery (encoding, legal actions, transitions)               #
    # ----------------------------------------------------------------- #
    def _mdp_encode(self, state):
        elevators_t, persons_t, _ = state
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
            p = self._elev_p[eidx]
            ef_s = list(ef); ef_s[eidx] = target
            outs = [(p, (tuple(ef_s), pl), 0.0)]
            others_set = set(self.elev_reachable[eidx]) - {target}
            others_set.add(ef[eidx])
            if others_set:
                pf = (1.0 - p) / len(others_set)
                for f in sorted(others_set):
                    ef_f = list(ef); ef_f[eidx] = f
                    outs.append((pf, (tuple(ef_f), pl), 0.0))
            return outs
        if kind == 'ENTER':
            _, pidx, eidx = action
            q = self._person_p[pidx]
            pl_s = list(pl); pl_s[pidx] = -eidx - 1
            return [(q, (ef, tuple(pl_s)), 0.0),
                    (1.0 - q, state, 0.0)]
        _, pidx, eidx = action
        q = self._person_p[pidx]
        floor = ef[eidx]
        if floor == self.person_goal[pidx]:
            pl_s = list(pl); pl_s[pidx] = DELIV
            reward = self._person_mean_r[pidx]
            if all(x == DELIV for x in pl_s):
                reward += self.goal_reward
                nxt = self._mdp_init_state
            else:
                nxt = (ef, tuple(pl_s))
            return [(q, nxt, reward), (1.0 - q, state, 0.0)]
        pl_s = list(pl); pl_s[pidx] = floor
        return [(q, (ef, tuple(pl_s)), 0.0),
                (1.0 - q, state, 0.0)]

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

    @staticmethod
    def _parse_action(action):
        if action == "RESET":
            return ("RESET", None)
        try:
            kind, args = action.split("{")
            args = args.rstrip("}").split(",")
            return kind, (int(args[0]), int(args[1]))
        except Exception:
            return None, None

"""
AI disclosure:
  Used: Claude Code for brainstorming pruning strategies and
  discussing admissibility arguments for the heuristic. The code
  was written by Claude. I (Shaked) directed the design through
  discussion, verified correctness against the specification, ran
  the local checker, and validated the implementation.
"""

import search
import utils
from collections import deque

id = ["208904839"]


# --------------------------------------------------------------------------- #
# Hashable State                                                              #
# --------------------------------------------------------------------------- #
class State:
    """
    Hashable state for the elevator problem.

    Fields:
      elev_floors  : tuple[int]   — floor of each elevator (in sorted-ID order)
      person_locs  : tuple[int]   — for each person (sorted-ID order):
                                       loc >= 0 -> standing on floor `loc`
                                       loc <  0 -> inside elevator with idx
                                                   eidx = -loc - 1
      _last_action : tuple|None   — identity of the action that produced this
                                    state. EXCLUDED from hash/eq so the closed
                                    set treats physically-equal states as
                                    equal. Used by successor() to skip
                                    immediate reversals:
                                       ('M', eidx, prev_floor)
                                       ('E', pidx, eidx)
                                       ('X', pidx, eidx)
                                        None  (initial state)
    """
    __slots__ = ("elev_floors", "person_locs", "_last_action", "_hash")

    def __init__(self, elev_floors, person_locs, last_action=None):
        self.elev_floors = elev_floors
        self.person_locs = person_locs
        self._last_action = last_action
        # Pre-compute hash once at construction. State is never mutated
        # after this, so the cached value is always correct.
        self._hash = hash((elev_floors, person_locs))

    def __eq__(self, other):
        # The closed set only ever holds State objects, so a State<->State
        # comparison is the only case we encounter.
        return (self.elev_floors == other.elev_floors
                and self.person_locs == other.person_locs)

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return "State(floors=%s, persons=%s)" % (self.elev_floors,
                                                 self.person_locs)


# --------------------------------------------------------------------------- #
# Problem                                                                      #
# --------------------------------------------------------------------------- #
class ElevatorsProblem(search.Problem):
    """Multi-elevator planning problem."""

    def __init__(self, initial):
        # ---- parse input dictionary -------------------------------------- #
        self.height = initial["height"]

        elevators = initial["Elevators"]
        persons = initial["Persons"]

        self.elevator_ids = sorted(elevators.keys())
        self.person_ids = sorted(persons.keys())

        # idx <-> id maps (idx is position in sorted list, used in tuples)
        self.eidx_of = {eid: i for i, eid in enumerate(self.elevator_ids)}
        self.pidx_of = {pid: i for i, pid in enumerate(self.person_ids)}

        # static elevator info (indexed by eidx)
        self.elev_reachable = []   # list[frozenset[int]]
        self.elev_capacity = []    # list[int]
        # An elevator's starting floor may not be in its reachable set
        # (one-way elevators). Include it so overlap/transitive computations
        # see the initial transfer opportunity.
        self.elev_reachable_plus_start = []
        for eid in self.elevator_ids:
            f0, reachable, wmax = elevators[eid]
            self.elev_reachable.append(frozenset(reachable))
            self.elev_reachable_plus_start.append(frozenset(reachable) | {f0})
            self.elev_capacity.append(wmax)

        # static person info (indexed by pidx)
        self.person_weight = []    # list[int]
        self.person_goal = []      # list[int]
        for pid in self.person_ids:
            f0, w, g = persons[pid]
            self.person_weight.append(w)
            self.person_goal.append(g)

        # Pre-compute the goal-state encoding of person_locs as a tuple.
        # In a goal state, every person stands on their goal floor (loc >= 0
        # equals goal_floor). person_locs == goal_locs_tuple iff goal reached
        # — a single C-level tuple compare in goal_test.
        self.goal_locs_tuple = tuple(self.person_goal)

        # ---- precompute static analyses --------------------------------- #
        # min_elev[f1][f2] = min number of distinct elevators needed to
        # travel from f1 to f2 (BFS in floor-connectivity graph).
        self._compute_min_elev()

        # transitive reachable set per elevator (for ENTER pruning)
        self._compute_transitive_reach()

        # useful exit floors per person (for EXIT pruning)
        self._compute_useful_exit_floors()

        # min_stints_in_elev[eidx][g] = min number of elevator-stints a
        # person needs given they are CURRENTLY inside elevator eidx and
        # heading to floor g. Independent of the elevator's current floor —
        # this is what makes the per-person heuristic component CONSISTENT
        # under MOVE actions (h_p does not change when the elevator moves).
        self._compute_min_stints_in_elev()

        # elev_overlap[(i, j)] = frozenset of floors reachable by BOTH
        # elevator i and elevator j (transfer-floor candidates). Precomputed
        # so the successor function avoids recomputing set intersections
        # for every state.
        self._compute_elev_overlap()

        # transfer_floors[eidx][g] = union, over every elevator OTHER than
        # eidx whose transitive reach includes g, of the overlap floors with
        # eidx. This lets the successor function answer "what floors should
        # E move to so a passenger heading to g can transfer?" via a single
        # dict lookup instead of an inner loop over all elevators.
        self._compute_transfer_floors()

        # ---- build initial state ---------------------------------------- #
        elev_floors = tuple(elevators[eid][0] for eid in self.elevator_ids)
        person_locs = tuple(persons[pid][0] for pid in self.person_ids)
        initial_state = State(elev_floors, person_locs, last_action=None)

        search.Problem.__init__(self, initial_state)

    # --------------------------------------------------------------------- #
    # Precomputations                                                       #
    # --------------------------------------------------------------------- #
    def _compute_min_elev(self):
        """
        Build the floor-connectivity graph (edge f1—f2 iff some elevator
        reaches both). Then BFS only from the floors we will ever query
        from: the set of distinct person-goal floors.

        For each goal g in `goal_set`, store a dict
            dist_to_goal[g] = { f: min number of elevators from f to g }
        Floors not reachable from g are simply absent (treated as +inf).

        Scales with O(|goals| * (F + edges)) instead of O(F^3).
        """
        relevant_floors = set()
        for reach in self.elev_reachable_plus_start:
            relevant_floors |= reach
        relevant_floors.update(self.person_goal)

        adj = {f: set() for f in relevant_floors}
        for reach in self.elev_reachable_plus_start:
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
        """
        For each elevator, the set of floors reachable from it via any
        chain of transfers (BFS over the elevator-overlap graph).
        """
        n = len(self.elevator_ids)
        # elevator-overlap graph
        e_adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if self.elev_reachable_plus_start[i] & self.elev_reachable_plus_start[j]:
                    e_adj[i].add(j)
                    e_adj[j].add(i)

        self.elev_transitive = []
        for i in range(n):
            visited = {i}
            q = deque([i])
            union = set(self.elev_reachable_plus_start[i])
            while q:
                u = q.popleft()
                for v in e_adj[u]:
                    if v not in visited:
                        visited.add(v)
                        union |= self.elev_reachable_plus_start[v]
                        q.append(v)
            self.elev_transitive.append(frozenset(union))

    def _compute_useful_exit_floors(self):
        """
        For each person p, useful_exit_floors[pidx] is the set of floors
        where exiting can lead to reaching the goal:
           goal floor itself, or
           a floor reachable by some elevator whose transitive set contains
           the goal.
        """
        self.useful_exit = []
        for pidx, pid in enumerate(self.person_ids):
            g = self.person_goal[pidx]
            useful = {g}
            for eidx2 in range(len(self.elevator_ids)):
                if g in self.elev_transitive[eidx2]:
                    useful |= self.elev_reachable_plus_start[eidx2]
            self.useful_exit.append(frozenset(useful))

    def _compute_min_stints_in_elev(self):
        """
        For each (elevator E, person-goal floor g):
            min_stints_in_elev[E][g] = 1                 if g in E.reach
                                     = 1 + min_{f' in E.reach} dist_to_goal[g][f']
                                                                otherwise
        Only computed for goals that actually appear in `Persons`, so this
        scales with O(E * |goals|) rather than O(E * F).

        Independent of E's current floor — crucial for heuristic
        consistency under MOVE.
        """
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
        """
        For every ordered pair (i, j) of distinct elevator indices, store
        the frozenset of floors reachable by BOTH. Used by the successor
        function to enumerate transfer-floor candidates without repeating
        set intersections at every state expansion.
        """
        n = len(self.elevator_ids)
        self.elev_overlap = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                self.elev_overlap[(i, j)] = (
                    self.elev_reachable[i] & self.elev_reachable_plus_start[j]
                )

    def _compute_transfer_floors(self):
        """
        transfer_floors[eidx][g] = set of floors elevator `eidx` can move to
        so that a passenger heading to `g` (whose goal is NOT in eidx.reach)
        can be handed off to some other elevator whose transitive reach
        contains `g`. Precomputing this collapses an O(E)-per-passenger inner
        loop in successor into a single dict lookup.

        Only computed for goals that appear in `Persons`.
        """
        n = len(self.elevator_ids)
        self.transfer_floors = [dict() for _ in range(n)]
        goal_set = set(self.person_goal)
        for eidx in range(n):
            for g in goal_set:
                # only relevant when g is NOT directly reachable by eidx
                if g in self.elev_reachable[eidx]:
                    continue
                acc = set()
                for other in range(n):
                    if other == eidx:
                        continue
                    if g in self.elev_transitive[other]:
                        acc |= self.elev_overlap[(eidx, other)]
                self.transfer_floors[eidx][g] = frozenset(acc)

    # --------------------------------------------------------------------- #
    # Helpers                                                                #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _loc_in_elev(loc):
        """True iff person location encodes 'inside an elevator'."""
        return loc < 0

    @staticmethod
    def _eidx_from_loc(loc):
        """Decode 'in elevator' loc into eidx."""
        return -loc - 1

    @staticmethod
    def _encode_in_elev(eidx):
        return -(eidx + 1)

    # --------------------------------------------------------------------- #
    # Required API: successor                                                #
    # --------------------------------------------------------------------- #
    def successor(self, state):
        elev_floors = state.elev_floors
        person_locs = state.person_locs
        last_action = state._last_action  # ('M', eidx, prev_floor) | ('E', pidx, eidx) | ('X', pidx, eidx) | None

        # Cache attribute lookups in locals (faster than self.X in tight loops).
        elev_ids = self.elevator_ids
        person_ids = self.person_ids
        person_goal = self.person_goal
        person_weight = self.person_weight
        elev_reachable = self.elev_reachable
        elev_capacity = self.elev_capacity
        elev_transitive = self.elev_transitive
        transfer_floors = self.transfer_floors
        useful_exit = self.useful_exit

        n_elev = len(elev_ids)
        n_pers = len(person_ids)

        # decode last_action so we can check it cheaply later
        la_kind = last_action[0] if last_action is not None else None

        # ---- one-pass classification ----------------------------------- #
        # on_floor_persons : list of (pidx, loc, goal) for persons standing
        # in_elev_persons[e]: list of (pidx, goal) for passengers of elevator e
        # elev_load[e]      : total weight inside elevator e
        on_floor_persons = []
        in_elev_persons = [[] for _ in range(n_elev)]
        elev_load = [0] * n_elev
        for pidx in range(n_pers):
            loc = person_locs[pidx]
            g = person_goal[pidx]
            if loc < 0:                              # inside an elevator
                eidx = -loc - 1
                in_elev_persons[eidx].append((pidx, g))
                elev_load[eidx] += person_weight[pidx]
            else:                                    # standing on a floor
                on_floor_persons.append((pidx, loc, g))

        # an elevator with a passenger AT its current floor must EXIT first
        elev_must_exit = [False] * n_elev
        for eidx in range(n_elev):
            ef = elev_floors[eidx]
            for _, g in in_elev_persons[eidx]:
                if g == ef:
                    elev_must_exit[eidx] = True
                    break

        successors = []

        # ----- MOVE actions ------------------------------------------------ #
        for eidx in range(n_elev):
            if elev_must_exit[eidx]:
                continue
            # Strong MOVE-cycle pruning: two consecutive MOVEs of the same
            # elevator with no ENTER/EXIT between them are dominated by a
            # single MOVE to the final destination (each MOVE = 1 action,
            # reaches any reachable floor). So skip ALL MOVE generation for
            # this elevator if the most recent action was its own MOVE.
            if la_kind == 'M' and last_action[1] == eidx:
                continue
            cur_floor = elev_floors[eidx]
            reach = elev_reachable[eidx]
            elev_trans = elev_transitive[eidx]

            candidates = set()

            # pickup: floors of on-floor persons (not at goal) reachable by E,
            # AND whose goal is in E's transitive reach. If g not in
            # transitive, ENTER would be rejected anyway, so MOVE-to-pickup
            # is wasted (R1 — deferred-form argument).
            for _, loc, g in on_floor_persons:
                if loc != g and loc in reach and g in elev_trans:
                    candidates.add(loc)

            # delivery / transfer for own passengers
            tf_for_e = transfer_floors[eidx]
            for _, g in in_elev_persons[eidx]:
                if g in reach:
                    candidates.add(g)
                else:
                    # precomputed: floors E can move to to set up a transfer
                    candidates |= tf_for_e[g]

            candidates.discard(cur_floor)

            eid = elev_ids[eidx]
            prefix = elev_floors[:eidx]
            suffix = elev_floors[eidx + 1:]
            for target in candidates:
                new_floors = prefix + (target,) + suffix
                new_state = State(
                    new_floors,
                    person_locs,
                    last_action=('M', eidx, cur_floor),
                )
                successors.append(("MOVE{%d,%d}" % (eid, target), new_state))

        # ----- ENTER actions ---------------------------------------------- #
        for pidx, loc, g in on_floor_persons:
            if loc == g:
                continue                              # already at goal
            pid = person_ids[pidx]
            w = person_weight[pidx]
            pre = person_locs[:pidx]
            post = person_locs[pidx + 1:]
            for eidx in range(n_elev):
                if elev_floors[eidx] != loc:
                    continue
                # Force EXIT before ENTER when this elevator has any
                # passenger AT their goal floor: any optimal plan can be
                # rearranged to do the at-goal EXIT first (proved by the
                # swap-commute argument; ENTER and EXIT of different persons
                # at the same floor commute, and capacity in the swapped
                # plan is no stricter).
                if elev_must_exit[eidx]:
                    continue
                if g not in elev_transitive[eidx]:
                    continue
                if elev_load[eidx] + w > elev_capacity[eidx]:
                    continue
                # block immediate ENTER undoing the previous EXIT:
                # last_action ('X', pidx, eidx) means p just exited E here,
                # re-entering would put us back at the pre-EXIT state.
                if la_kind == 'X' and last_action[1] == pidx and last_action[2] == eidx:
                    continue
                encoded = -eidx - 1
                new_locs = pre + (encoded,) + post
                new_state = State(
                    elev_floors,
                    new_locs,
                    last_action=('E', pidx, eidx),
                )
                successors.append(
                    ("ENTER{%d,%d}" % (pid, elev_ids[eidx]), new_state)
                )

        # ----- EXIT actions ----------------------------------------------- #
        for eidx in range(n_elev):
            ef = elev_floors[eidx]
            eid = elev_ids[eidx]
            for pidx, g in in_elev_persons[eidx]:
                if ef not in useful_exit[pidx]:
                    continue
                # block immediate EXIT undoing the previous ENTER:
                # last_action ('E', pidx, eidx) means p just entered E here,
                # exiting now puts p back on the same floor → identical to
                # pre-ENTER state.
                if la_kind == 'E' and last_action[1] == pidx and last_action[2] == eidx:
                    continue
                new_locs = (
                    person_locs[:pidx] + (ef,) + person_locs[pidx + 1:]
                )
                new_state = State(
                    elev_floors,
                    new_locs,
                    last_action=('X', pidx, eidx),
                )
                successors.append(
                    ("EXIT{%d,%d}" % (person_ids[pidx], eid), new_state)
                )

        return successors

    # --------------------------------------------------------------------- #
    # Required API: goal_test                                                #
    # --------------------------------------------------------------------- #
    def goal_test(self, state):
        # Single tuple compare. Correct because:
        #   - person on goal floor:   loc = goal_floor  -> equal
        #   - person inside elevator: loc < 0 (encoded), goal_floor >= 0 -> not equal
        #   - person on non-goal floor: loc != goal_floor -> not equal
        # All three goal conditions reduce to person_locs == goal_locs_tuple.
        return state.person_locs == self.goal_locs_tuple

    # --------------------------------------------------------------------- #
    # Required API: h_astar                                                  #
    # --------------------------------------------------------------------- #
    def h_astar(self, node):
        """
        Admissible AND consistent heuristic.

        Sums two disjoint lower bounds (different action types -> no overlap):

        (1) Per-person ENTER+EXIT lower bound:
              - on floor at goal           : 0
              - on floor not at goal       : 2 * min_elev[f][g]
              - in elevator E              : 2 * min_stints_in_elev[E][g] - 1
            (min_stints_in_elev is independent of E's current floor, which
            keeps the heuristic CONSISTENT across MOVE actions.)

        (2) Per-elevator delivery-MOVE lower bound:
              for each elevator E, count distinct passenger goals that are
              reachable by E and differ from E's current floor.

        Admissibility: each ENTER and EXIT is a separate, non-shared action
        belonging to exactly one person; each delivery-MOVE is a distinct
        floor-visit by exactly one elevator. No double counting.

        Consistency (h(s) <= 1 + h(s')) under each action:
          MOVE   : per-person component unchanged (min_stints independent
                   of E.floor); delivery set changes by at most 1 entry
                   (one floor per MOVE). |Delta h| <= 1.
          ENTER  : per-person h_p drops by exactly 1 in the best case (when
                   entering an optimal first elevator); delivery may rise by
                   at most 1; net change >= 0.
          EXIT   : per-person h_p increases (or matches) by 1 when exiting
                   at an optimal floor; delivery loses at most 1; net change
                   >= 0 in the best case.
        """
        state = node.state
        elev_floors = state.elev_floors
        person_locs = state.person_locs

        # Cache attribute lookups in locals (faster than self.X in tight loops).
        person_goal = self.person_goal
        dist_to_goal = self.dist_to_goal
        min_stints = self.min_stints_in_elev
        elev_reach = self.elev_reachable

        INF = 10 ** 9
        h = 0
        delivery_pairs = set()
        delivery_floors = set()
        pickup_floors = set()
        elev_floor_set = set(elev_floors)
        n_delivered = 0

        for pidx in range(len(person_locs)):
            loc = person_locs[pidx]
            g = person_goal[pidx]

            if loc >= 0:
                if loc == g:
                    n_delivered += 1
                    continue
                d = dist_to_goal[g].get(loc, INF)
                if d >= INF:
                    return INF
                h += 2 * d
                if loc not in elev_floor_set:
                    pickup_floors.add(loc)
            else:
                eidx = -loc - 1
                stints = min_stints[eidx].get(g, INF)
                if stints >= INF:
                    return INF
                h += 2 * stints - 1
                if g in elev_reach[eidx] and elev_floors[eidx] != g:
                    delivery_pairs.add((eidx, g))
                    delivery_floors.add(g)

        extra_pickup = len(pickup_floors - delivery_floors)
        return h + len(delivery_pairs) + extra_pickup - 0.0001 * n_delivered


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #
def create_elevators_problem(game):
    print("<<create_elevators_problem")
    return ElevatorsProblem(game)


if __name__ == '__main__':
    import ex1_check
    ex1_check.main()

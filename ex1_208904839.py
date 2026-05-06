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
      _last_move   : tuple|None   — (eidx, prev_floor) of the last MOVE since
                                    the last ENTER/EXIT. EXCLUDED from hash/eq
                                    so the closed set treats physically-equal
                                    states as equal.
    """
    __slots__ = ("elev_floors", "person_locs", "_last_move")

    def __init__(self, elev_floors, person_locs, last_move=None):
        self.elev_floors = elev_floors
        self.person_locs = person_locs
        self._last_move = last_move

    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return (self.elev_floors == other.elev_floors
                and self.person_locs == other.person_locs)

    def __hash__(self):
        return hash((self.elev_floors, self.person_locs))

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
        for eid in self.elevator_ids:
            f0, reachable, wmax = elevators[eid]
            self.elev_reachable.append(frozenset(reachable))
            self.elev_capacity.append(wmax)

        # static person info (indexed by pidx)
        self.person_weight = []    # list[int]
        self.person_goal = []      # list[int]
        for pid in self.person_ids:
            f0, w, g = persons[pid]
            self.person_weight.append(w)
            self.person_goal.append(g)

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

        # ---- build initial state ---------------------------------------- #
        elev_floors = tuple(elevators[eid][0] for eid in self.elevator_ids)
        person_locs = tuple(persons[pid][0] for pid in self.person_ids)
        initial_state = State(elev_floors, person_locs, last_move=None)

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
        # only build adjacency over floors that any elevator actually
        # touches (irrelevant floors never appear in any state we'll query)
        relevant_floors = set()
        for reach in self.elev_reachable:
            relevant_floors |= reach
        # also include person start floors and goals so lookups never miss
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
        """
        For each elevator, the set of floors reachable from it via any
        chain of transfers (BFS over the elevator-overlap graph).
        """
        n = len(self.elevator_ids)
        # elevator-overlap graph
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
                    useful |= self.elev_reachable[eidx2]
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
        last_move = state._last_move

        n_elev = len(self.elevator_ids)
        n_pers = len(self.person_ids)

        # which persons sit in which elevator + load per elevator
        elev_load = [0] * n_elev
        passengers_of = [[] for _ in range(n_elev)]   # list of pidx
        for pidx in range(n_pers):
            loc = person_locs[pidx]
            if self._loc_in_elev(loc):
                eidx = self._eidx_from_loc(loc)
                elev_load[eidx] += self.person_weight[pidx]
                passengers_of[eidx].append(pidx)

        # an elevator with a passenger at this elevator's current floor
        # MUST exit before moving (pruning rule A).
        elev_must_exit = [False] * n_elev
        for eidx in range(n_elev):
            ef = elev_floors[eidx]
            for pidx in passengers_of[eidx]:
                if self.person_goal[pidx] == ef:
                    elev_must_exit[eidx] = True
                    break

        successors = []

        # ----- MOVE actions ------------------------------------------------ #
        for eidx in range(n_elev):
            if elev_must_exit[eidx]:
                continue  # must EXIT first — skip MOVE generation entirely
            cur_floor = elev_floors[eidx]
            reach = self.elev_reachable[eidx]

            # build candidate floor set (relevance pruning)
            candidates = set()

            for pidx in range(n_pers):
                loc = person_locs[pidx]
                g = self.person_goal[pidx]

                if not self._loc_in_elev(loc):
                    # person on floor `loc`
                    if loc != g and loc in reach:
                        candidates.add(loc)         # potential pickup
                    if loc != g and g in reach:
                        candidates.add(g)            # anticipatory dropoff floor
                else:
                    # person inside some elevator
                    in_eidx = self._eidx_from_loc(loc)
                    if in_eidx == eidx:
                        # this elevator's own passenger
                        if g in reach:
                            candidates.add(g)        # delivery
                        else:
                            # transfer floors with elevators that can reach g
                            for other in range(n_elev):
                                if other == eidx:
                                    continue
                                if g in self.elev_transitive[other]:
                                    overlap = reach & self.elev_reachable[other]
                                    candidates |= overlap
                    else:
                        # passenger of another elevator that needs transfer:
                        # we may anticipatorily position to receive them.
                        if g not in self.elev_reachable[in_eidx] \
                                and g in self.elev_transitive[eidx]:
                            overlap = reach & self.elev_reachable[in_eidx]
                            candidates |= overlap

            candidates.discard(cur_floor)

            # block immediate move-back (rule F via _last_move)
            if last_move is not None and last_move[0] == eidx:
                candidates.discard(last_move[1])

            eid = self.elevator_ids[eidx]
            for target in candidates:
                new_floors = list(elev_floors)
                new_floors[eidx] = target
                new_state = State(
                    tuple(new_floors),
                    person_locs,
                    last_move=(eidx, cur_floor),
                )
                action = "MOVE{%d,%d}" % (eid, target)
                successors.append((action, new_state))

        # ----- ENTER actions ---------------------------------------------- #
        for pidx in range(n_pers):
            loc = person_locs[pidx]
            if self._loc_in_elev(loc):
                continue
            g = self.person_goal[pidx]
            if loc == g:
                continue                              # already at goal
            pid = self.person_ids[pidx]
            w = self.person_weight[pidx]

            for eidx in range(n_elev):
                if elev_floors[eidx] != loc:
                    continue
                # transitive reachability: this elevator must be able to
                # eventually deliver the person
                if g not in self.elev_transitive[eidx]:
                    continue
                if elev_load[eidx] + w > self.elev_capacity[eidx]:
                    continue

                new_locs = list(person_locs)
                new_locs[pidx] = self._encode_in_elev(eidx)
                new_state = State(
                    elev_floors,
                    tuple(new_locs),
                    last_move=None,                  # ENTER resets the lock
                )
                eid = self.elevator_ids[eidx]
                action = "ENTER{%d,%d}" % (pid, eid)
                successors.append((action, new_state))

        # ----- EXIT actions ----------------------------------------------- #
        for pidx in range(n_pers):
            loc = person_locs[pidx]
            if not self._loc_in_elev(loc):
                continue
            eidx = self._eidx_from_loc(loc)
            ef = elev_floors[eidx]
            # only exit at useful floors (rule C)
            if ef not in self.useful_exit[pidx]:
                continue

            new_locs = list(person_locs)
            new_locs[pidx] = ef
            new_state = State(
                elev_floors,
                tuple(new_locs),
                last_move=None,                      # EXIT resets the lock
            )
            pid = self.person_ids[pidx]
            eid = self.elevator_ids[eidx]
            action = "EXIT{%d,%d}" % (pid, eid)
            successors.append((action, new_state))

        return successors

    # --------------------------------------------------------------------- #
    # Required API: goal_test                                                #
    # --------------------------------------------------------------------- #
    def goal_test(self, state):
        person_locs = state.person_locs
        for pidx in range(len(self.person_ids)):
            loc = person_locs[pidx]
            if self._loc_in_elev(loc):
                return False
            if loc != self.person_goal[pidx]:
                return False
        return True

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

        n_elev = len(self.elevator_ids)
        INF = 10 ** 9
        h = 0
        delivery = [set() for _ in range(n_elev)]

        for pidx in range(len(self.person_ids)):
            loc = person_locs[pidx]
            g = self.person_goal[pidx]

            if not self._loc_in_elev(loc):
                if loc == g:
                    continue
                # .get() so an unreachable (loc -> g) yields INF instead of
                # KeyError. INF propagates to h, signalling unreachability.
                d = self.dist_to_goal[g].get(loc, INF)
                if d >= INF:
                    return INF
                h += 2 * d
            else:
                eidx = self._eidx_from_loc(loc)
                ef = elev_floors[eidx]
                # use min_stints_in_elev (independent of ef) for consistency
                stints = self.min_stints_in_elev[eidx].get(g, INF)
                if stints >= INF:
                    return INF
                h += 2 * stints - 1
                # contribute to delivery set only if E can deliver directly
                # AND is not already at the goal floor
                if g in self.elev_reachable[eidx] and ef != g:
                    delivery[eidx].add(g)

        for eidx in range(n_elev):
            h += len(delivery[eidx])

        return h


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #
def create_elevators_problem(game):
    print("<<create_elevators_problem")
    return ElevatorsProblem(game)


if __name__ == '__main__':
    import ex1_check
    ex1_check.main()

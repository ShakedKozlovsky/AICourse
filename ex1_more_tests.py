import time
import ex1
import search as search
import utils as utils

def run_problem(func, targs=(), kwargs=None):
    if kwargs is None:
        kwargs = {}
    result = (-3, "default")
    try:
        result = func(*targs, **kwargs)
    except Exception as e:
        result = (-3, e)
    return result


def solve_problems(problem, expected_optimal=None):
    try:
        p = ex1.create_elevators_problem(problem)
    except Exception as e:
        return False, False, "ERROR", None

    result = run_problem((lambda p: search.astar_search(p, p.h_astar)), targs=[p])

    if result:
        if isinstance(result[0], search.Node):
            solve = result[0].path()[::-1]
            solution = [pi.action for pi in solve][1:]
            length = len(solution)

            solved_ok = True
            optimal_ok = (expected_optimal is not None and length == expected_optimal)

            if expected_optimal is None:
                status = "SOLVED"
            elif optimal_ok:
                status = "OPTIMAL"
            else:
                status = f"SUBOPTIMAL (expected {expected_optimal})"

            return solved_ok, optimal_ok, status, length
        else:
            if expected_optimal == 0:
                return True, True, "NO-SOLUTION (EXPECTED)", None
            return False, False, "NO-SOLUTION (UNEXPECTED)", None
    else:
        if expected_optimal == 0:
            return True, True, "NO-SOLUTION (EXPECTED)", None
        else:            
            return False, False, "FAILED", None


# Format reminder:
# {
#   "height":   number,
#   "Elevators":  {eid: (current_floor, allowed_floors, max_weight), ...},
#   "Persons":   {pid: (current_floor, weight, goal_floor), ...},
# }

# ── Original ─────────────────────────────────────────────────────────────────

#Optimal: 13
init_state_p1 = {
    "height": 6,
    "Elevators": {
        0: (0, (0, 1, 2, 3), 8),
        1: (4, (2, 4, 5, 6), 10)
    },
    "Persons": {
        10: (0, 3, 3),
        11: (2, 4, 6),
        12: (4, 5, 0)
    }
}

# ── Easy (similar difficulty to p1: 2-3 persons, 2 elevators) ────────────────

#Optimal: 10
# E1: 3 persons, 2 full-range elevators, no transfers needed
init_state_e1 = {
    "height": 5,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4, 5), 15),
        1: (5, (0, 1, 2, 3, 4, 5), 15)
    },
    "Persons": {
        10: (0, 3, 5),
        11: (5, 3, 0),
        12: (3, 3, 1)
    }
}

#Optimal: 11
# E2: 3 persons, 2 elevators with non-overlapping ranges, each person served directly
init_state_e2 = {
    "height": 6,
    "Elevators": {
        0: (0, (0, 1, 2, 3), 10),
        1: (6, (3, 4, 5, 6), 10)
    },
    "Persons": {
        10: (1, 3, 3),
        11: (5, 3, 4),
        12: (0, 3, 2)
    }
}

#Optimal: 18
# E3: 3 persons, 2 overlapping elevators, 1 person needs a transfer (like p1)
init_state_e3 = {
    "height": 6,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 10),
        1: (6, (2, 4, 5, 6), 10)
    },
    "Persons": {
        10: (0, 3, 5),
        11: (6, 3, 1),
        12: (3, 4, 6)
    }
}

#Optimal: 9
# E4: 3 persons, 2 elevators, weight capacity forces one-at-a-time
init_state_e4 = {
    "height": 5,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4, 5), 7),
        1: (5, (0, 1, 2, 3, 4, 5), 7)
    },
    "Persons": {
        10: (0, 5, 5),
        11: (0, 5, 3),
        12: (5, 5, 0)
    }
}

#Optimal: 13
# E5: 3 persons, 2 elevators with overlap, 1 transfer required
init_state_e5 = {
    "height": 7,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 12),
        1: (7, (3, 4, 5, 6, 7), 12)
    },
    "Persons": {
        10: (1, 4, 3),
        11: (6, 4, 7),
        12: (0, 4, 7)
    }
}

# ── Medium (4-5 persons, more transfers and coordination) ─────────────────────

#Optimal: 24
# M1: 4 persons, 2 elevators, all need transfers
init_state_m1 = {
    "height": 6,
    "Elevators": {
        0: (0, (0, 1, 2, 3,), 12),
        1: (6, (3, 4, 5, 6,), 12)
    },
    "Persons": {
        10: (0, 4, 5),
        11: (5, 4, 0),
        12: (2, 4, 6),
        13: (5, 4, 1)
    }
}

#Optimal: 21
# M2: 4 persons, 3 elevators, bypass elevator available
init_state_m2 = {
    "height": 8,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 10),
        1: (4, (2, 4, 6, 8), 10)
    },
    "Persons": {
        10: (0, 3, 8),
        11: (8, 3, 0),
        12: (2, 3, 6),
        13: (6, 3, 1)
    }
}

#Optimal: 22
# M3: 4 persons, 3 elevators, bypass elevator available
init_state_m3 = {
    "height": 8,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 10),
        1: (4, (2, 4, 6, 8), 10),
        2: (4, (7,2), 10)
    },
    "Persons": {
        10: (0, 3, 8),
        11: (8, 3, 0),
        12: (2, 3, 7),
        13: (6, 3, 1)
    }
}

#Optimal: 16
# M4: 5 persons, 2 elevators, weight forces single occupancy
init_state_m4 = {
    "height": 6,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4, 5, 6), 8),
        1: (6, (0, 1, 2, 3, 4, 5, 6), 8)
    },
    "Persons": {
        10: (0, 5, 6),
        11: (0, 5, 4),
        12: (6, 5, 0),
        13: (6, 5, 2),
        14: (3, 5, 6)
    }
}


#Optimal: 25
# M5: 4 persons, 2 full-range elevators, weight allows only 1 person each
init_state_m5 = {
    "height": 8,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 10),
        1: (4, (4, 5, 6, 7, 8), 10)
    },
    "Persons": {
        10: (0, 6, 8),
        11: (0, 4, 5),
        12: (8, 6, 0),
        13: (8, 5, 3)
    }
}

# ── Custom additional tests ───────────────────────────────────────────────────

# optimal: 7
# C1: 2 persons, 2 full-range elevators, direct service with little branching
init_state_c1 = {
    "height": 4,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 10),
        1: (4, (0, 1, 2, 3, 4), 10)
    },
    "Persons": {
        10: (1, 2, 3),
        11: (3, 2, 0)
    }
}

# optimal: 15
# C2: 3 persons, transfer at floor 2 or 4 depending on route choice
init_state_c2 = {
    "height": 6,
    "Elevators": {
        0: (0, (0, 1, 2, 3), 9),
        1: (6, (2, 4, 5, 6), 9)
    },
    "Persons": {
        10: (0, 3, 6),
        11: (6, 3, 1),
        12: (2, 2, 5)
    }
}

# optimal: 21
# C3: 4 persons, asymmetric ranges to exercise the pruning logic
init_state_c3 = {
    "height": 7,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 11),
        1: (7, (3, 4, 5, 6, 7), 11)
    },
    "Persons": {
        10: (0, 4, 7),
        11: (1, 3, 6),
        12: (6, 3, 0),
        13: (7, 2, 3)
    }
}

# easy solution:
# E6: 2 persons, fully connected elevators, direct round-trip
init_state_e6 = {
    "height": 4,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 10),
        1: (4, (0, 1, 2, 3, 4), 10)
    },
    "Persons": {
        20: (0, 2, 4),
        21: (4, 2, 0)
    }
}

# easy solution:
# E7: 2 persons, one elevator can do the whole job with no transfer
init_state_e7 = {
    "height": 5,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4, 5), 12),
        1: (5, (0, 1, 2, 3, 4, 5), 12)
    },
    "Persons": {
        22: (1, 3, 5),
        23: (5, 3, 1)
    }
}

# medium solution:
# M6: 3 persons, transfer through a shared middle floor
init_state_m6 = {
    "height": 7,
    "Elevators": {
        0: (0, (0, 1, 2, 3), 10),
        1: (7, (3, 4, 5, 6, 7), 10)
    },
    "Persons": {
        30: (0, 4, 7),
        31: (7, 4, 0),
        32: (2, 3, 6)
    }
}

# medium solution:
# M7: 4 persons, 3 elevators with a short transfer chain
init_state_m7 = {
    "height": 8,
    "Elevators": {
        0: (0, (0, 1, 2, 3), 9),
        1: (4, (3, 4, 5, 6), 9),
        2: (8, (6, 7, 8), 9)
    },
    "Persons": {
        33: (0, 3, 8),
        34: (8, 3, 0),
        35: (2, 2, 6),
        36: (6, 2, 1)
    }
}

# hard solution:
# H1: 4 persons, three elevators, one person needs two transfers
init_state_h1 = {
    "height": 9,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 14),
        1: (4, (3, 4, 5, 6, 7), 14),
        2: (9, (7, 8, 9), 14)
    },
    "Persons": {
        40: (0, 4, 9),
        41: (9, 4, 0),
        42: (2, 2, 7),
        43: (7, 2, 1)
    }
}

# hard solution:
# H2: 5 persons, crowded transfer floors with asymmetric elevator reach
init_state_h2 = {
    "height": 10,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4, 5), 16),
        1: (5, (4, 5, 6, 7, 8), 16),
        2: (10, (8, 9, 10), 16)
    },
    "Persons": {
        44: (0, 3, 10),
        45: (10, 3, 0),
        46: (2, 3, 8),
        47: (8, 3, 2),
        48: (5, 3, 7)
    }
}

# extra hard solution:
# XH1: 5 persons, two transfer chains plus mixed capacities
init_state_xh1 = {
    "height": 12,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 20),
        1: (4, (3, 4, 5, 6, 7), 12),
        2: (8, (7, 8, 9, 10, 11, 12), 20)
    },
    "Persons": {
        50: (0, 5, 12),
        51: (12, 5, 0),
        52: (2, 5, 9),
        53: (9, 5, 2),
        54: (4, 4, 8)
    }
}

# extra hard solution:
# XH2: 6 persons, longer chain with a bottleneck elevator
init_state_xh2 = {
    "height": 12,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 15),
        1: (4, (4, 5, 6, 7, 8), 8),
        2: (8, (8, 9, 10, 11, 12), 15)
    },
    "Persons": {
        55: (0, 3, 12),
        56: (12, 3, 0),
        57: (2, 3, 8),
        58: (8, 3, 2),
        59: (4, 2, 11),
        60: (11, 2, 4)
    }
}

#easy solution: 
test_case_bottleneck = {
    "height": 5,
    "Elevators": {
        0: (0, (0, 1, 2), 500),    # Limited to lower floors
        1: (0, (0, 1, 2, 3, 4, 5), 500) # Can reach all floors
    },
    "Persons": {
        1: (0, 80, 5),  # Needs Elevator 1
        2: (2, 70, 0)   # Can use either, but may block Elevator 1
    }
}

# medium solution:
test_case_weight = {
    "height": 4,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 100) # Max capacity 100kg
    },
    "Persons": {
        1: (0, 60, 4),
        2: (0, 60, 3) # Combined weight (120) exceeds capacity (100)
    }
}

#medium solution:
test_case_relay = {
    "height": 10,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4, 5), 200),     # Only lower half
        1: (10, (5, 6, 7, 8, 9, 10), 200)    # Only upper half
    },
    "Persons": {
        100: (0, 75, 10) # Must "transfer" at floor 5
    }
}

#medium solution:
test_case_divided = {
    "height": 6,
    "Elevators": {
        0: (0, (0, 1, 2, 3), 200),  # Services ground to middle
        1: (6, (3, 4, 5, 6), 200)   # Services middle to top
    },
    "Persons": {
        1: (0, 70, 6) # Must be dropped at floor 3 and picked up by E1
    }
}

#medium solution:
test_case_handoff = {
    "height": 4,
    "Elevators": {
        0: (0, (0, 1, 2), 1000), # Heavy capacity, limited floors
        1: (4, (2, 3, 4), 1000), # Heavy capacity, limited floors
        2: (2, (0, 1, 2, 3, 4), 100) # Full reach, but too weak for Person 1
    },
    "Persons": {
        1: (0, 500, 4) # Weight 500kg forces the use of E0 then E1
    }
}

#medium solution:
test_case_crowded = {
    "height": 10,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 250)
    },
    "Persons": {
        1: (0, 80, 10),
        2: (0, 80, 2),
        3: (0, 80, 5)
    }
}

#medium solution:
test_case_express = {
    "height": 8,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8), 200),
        1: (0, (0, 4, 8), 200) # Express: only 0, 4, and 8
    },
    "Persons": {
        1: (0, 70, 8), # Should take the Express
        2: (0, 70, 3)  # Must take the Local
    }
}

#medium-hard solution:
test_case_weight_relay = {
    "height": 8,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 1000),  # Heavy reach (low)
        1: (8, (4, 5, 6, 7, 8), 500)    # Medium reach (high)
    },
    "Persons": {
        30: (0, 450, 8), # Heavy: Must transfer at Floor 4
        31: (0, 100, 8)  # Light: Must also transfer at Floor 4
    }
}

#medium-hard solution:
test_case_shuffle = {
    "height": 10,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4, 5), 300),
        1: (5, (2, 3, 4, 5, 6, 7, 8), 300),
        2: (10, (5, 6, 7, 8, 9, 10), 300)
    },
    "Persons": {
        40: (0, 70, 10), # Must use E0 -> E1 -> E2 (Two transfers!)
        41: (2, 70, 8),  # Must use E1 -> E2
        42: (10, 70, 0)  # Must use E2 -> E1 -> E0
    }
}

# hard solution:
test_case_apartment = {
    "height": 8,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8), 200),
        1: (8, (0, 2, 4, 6, 8), 300), # Express elevator (even floors only)
        2: (4, (0, 1, 2, 3, 4), 150)
    },
    "Persons": {
        1: (1, 70, 7),
        2: (7, 90, 0),
        3: (0, 110, 8),
        4: (4, 80, 2),
        5: (2, 65, 6)
    }
}

#hard solution:
test_case_heavy = {
    "height": 12,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4, 5, 6), 1000), # High capacity, low reach
        1: (12, (6, 7, 8, 9, 10, 11, 12), 1000), # High capacity, high reach
        2: (6, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 150) # Full reach, low capacity
    },
    "Persons": {
        10: (1, 900, 12), # Must transfer at floor 6, Elevator 2 cannot pick them up
        11: (11, 100, 1), # Can use any, but will likely use Elevator 2 or a transfer
        12: (6, 80, 0),
        13: (6, 80, 12),
        14: (0, 50, 12)
    }
}

init_state_no_solution_reach = {
    "height": 8,
    "Elevators": {
        0: (0, (0, 1, 2, 3), 200),
        1: (8, (5, 6, 7, 8), 200)
    },
    "Persons": {
        10: (0, 70, 8)
    }
}

init_state_no_solution_weight = {
    "height": 8,
    "Elevators": {
        0: (0, (0, 1, 2), 200),
        1: (4, (3, 4, 5), 50) 
    },
    "Persons": {
        10: (4, 100, 5)
    }
}

init_state_no_solution_goal_access = {
    "height": 8,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 200),
        1: (4, (4, 5, 6, 7), 200)
    },
    "Persons": {
        10: (0, 70, 8)
    }
}






def main():
    start = time.time()
    problems = [
        ("p1  (medium)", init_state_p1, 13),
        ("e1  (easy)",     init_state_e1, 10),
        ("e2  (easy)",     init_state_e2, 11),
        ("e3  (easy)",     init_state_e3, 18),
        ("e4  (easy)",     init_state_e4, 9),
        ("e5  (easy)",     init_state_e5, 13),
        ("m1  (medium)",   init_state_m1, 24),
        ("m2  (medium)",   init_state_m2, 21),
        ("m3  (medium)",   init_state_m3, 22),
        ("m4  (medium)",   init_state_m4, 16),
        ("m5  (medium)",   init_state_m5, 25),
        ("c1  (easy)", init_state_c1, 7),
        ("c2  (easy)", init_state_c2, 15),
        ("c3  (medium)", init_state_c3, 21),
        ("e6  (easy)", init_state_e6, 6),
        ("e7  (easy)", init_state_e7, 6),
        ("m6  (medium)", init_state_m6, 18),
        ("m7  (medium)", init_state_m7, 29),
        ("bottleneck  (easy)", test_case_bottleneck, 7),
        ("weight  (easy)", test_case_weight, 7),
        ("relay  (easy)", test_case_relay, 7),
        ("divided  (easy)", test_case_divided, 7),
        ("handoff  (easy)", test_case_handoff, 7),
        ("crowded  (easy)", test_case_crowded, 9),
        ("express  (easy)", test_case_express, 6),
        ("weight_relay  (easy)", test_case_weight_relay, 13),
        ("shuffle  (medium)", test_case_shuffle, 16),
        ("h1  (hard)", init_state_h1, 28),
        # ("h2  (apartment, too slow on planet)", test_case_apartment, 17),
        # ("xh1  (heavy, too slow)", test_case_heavy, 20),
        # ("xh2  (5p, 3e, too slow on planet)", init_state_h2, 31),
        # ("xxh1 (xh1 set, too slow)", init_state_xh1, 30),
        # ("xxh2 (xh2 set, too slow)", init_state_xh2, 30),
        ("no_solution_reach", init_state_no_solution_reach, 0),
        ("no_solution_weight", init_state_no_solution_weight, 0),
        ("no_solution_goal_access", init_state_no_solution_goal_access, 0),
    ]

    solved_count = 0
    optimal_count = 0
    result_rows = []

    for name, p, expected_optimal in problems:
        t0 = time.time()
        solved_ok, optimal_ok, status, length = solve_problems(p, expected_optimal)
        elapsed = time.time() - t0
        solved_count += int(solved_ok)
        optimal_count += int(optimal_ok)
        result_rows.append((name, solved_ok, optimal_ok, length, status, elapsed))

    end = time.time()
    total = len(problems)
    print(f"Solved: {solved_count}/{total}")
    print(f"Optimal: {optimal_count}/{total}")

    print("\nResults summary:")
    print(f"{'Test':<24} {'Solved':<8} {'Optimal':<8} {'Length':<8} {'Status':<30} {'Seconds':>8}")
    print(f"{'-' * 24} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 30} {'-' * 8}")
    for name, solved_ok, optimal_ok, length, status, elapsed in result_rows:
        length_str = '-' if length is None else str(length)
        print(f"{name:<24} {str(solved_ok):<8} {str(optimal_ok):<8} {length_str:<8} {status:<30} {elapsed:>8.2f}")

    print('All tests together took:', end - start, 'seconds.')


if __name__ == '__main__':
    main()

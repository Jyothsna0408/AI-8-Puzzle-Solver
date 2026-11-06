"""
Automated Comprehensive Verification Script for AI-Based 8-Puzzle Solver.
Tests all requirements specified in Section 9:
1. A puzzle requiring two moves.
2. An already-solved puzzle.
3. An unsolvable puzzle.
4. Invalid repeated numbers.
5. Random solvable puzzle.
6. Both heuristic comparison modes.
7. Verification of optimal solutions between both heuristics.
"""

import sys
import time
from solver import (
    GOAL_STATE,
    manhattan_distance,
    misplaced_tiles,
    count_inversions,
    is_solvable,
    validate_puzzle_input,
    solve_puzzle,
    generate_random_puzzle,
)


def run_all_tests():
    print("=" * 70)
    print("AI-BASED 8-PUZZLE GAME SOLVER - VERIFICATION SUITE")
    print("=" * 70)

    # -------------------------------------------------------------
    # Test 1: Puzzle requiring two moves
    # -------------------------------------------------------------
    print("\n[TEST 1] A puzzle requiring exactly two moves")
    two_move_state = (1, 2, 3, 4, 0, 6, 7, 5, 8)
    res_m = solve_puzzle(two_move_state, manhattan_distance, "Manhattan Distance")
    res_mis = solve_puzzle(two_move_state, misplaced_tiles, "Misplaced Tiles")

    print(f"  Initial State : {two_move_state}")
    print(f"  Manhattan moves: {res_m.moves_count}, Explored: {res_m.nodes_explored}, Time: {res_m.execution_time*1000:.2f} ms")
    print(f"  Actions taken  : {res_m.actions}")
    print(f"  Misplaced moves: {res_mis.moves_count}, Explored: {res_mis.nodes_explored}, Time: {res_mis.execution_time*1000:.2f} ms")

    assert res_m.moves_count == 2, f"Expected 2 moves, got {res_m.moves_count}"
    assert res_mis.moves_count == 2, f"Expected 2 moves, got {res_mis.moves_count}"
    assert res_m.path[-1] == GOAL_STATE, "Goal state not reached!"
    print("  >>> PASS: Both heuristics found the optimal 2-move solution!")

    # -------------------------------------------------------------
    # Test 2: An already-solved puzzle
    # -------------------------------------------------------------
    print("\n[TEST 2] An already-solved puzzle")
    solved_state = GOAL_STATE
    res_m_solved = solve_puzzle(solved_state, manhattan_distance, "Manhattan Distance")
    res_mis_solved = solve_puzzle(solved_state, misplaced_tiles, "Misplaced Tiles")

    print(f"  Input State   : {solved_state}")
    print(f"  Manhattan moves: {res_m_solved.moves_count}, Explored: {res_m_solved.nodes_explored}")
    print(f"  Misplaced moves: {res_mis_solved.moves_count}, Explored: {res_mis_solved.nodes_explored}")

    assert res_m_solved.moves_count == 0, "Already solved must be 0 moves"
    assert res_mis_solved.moves_count == 0, "Already solved must be 0 moves"
    assert res_m_solved.nodes_explored == 0, "Nodes explored should be 0"
    print("  >>> PASS: Handled instantly with 0 moves!")

    # -------------------------------------------------------------
    # Test 3: An unsolvable puzzle
    # -------------------------------------------------------------
    print("\n[TEST 3] An unsolvable puzzle (inversion count parity check)")
    # Swap 1 and 2 in goal state
    unsolvable_state = (2, 1, 3, 4, 5, 6, 7, 8, 0)
    inv = count_inversions(unsolvable_state)
    solvable = is_solvable(unsolvable_state)

    print(f"  Input State   : {unsolvable_state}")
    print(f"  Inversions    : {inv} (Odd parity)")
    print(f"  Is Solvable?  : {solvable}")

    assert inv % 2 == 1, "Inversions should be odd"
    assert not solvable, "Must be classified as unsolvable"
    print("  >>> PASS: Unsolvable state detected mathematically via odd inversions!")

    # -------------------------------------------------------------
    # Test 4: Invalid repeated numbers & boundary validations
    # -------------------------------------------------------------
    print("\n[TEST 4] Invalid input validation (repeated, out-of-range, missing)")
    dup_input = [1, 1, 2, 3, 4, 5, 6, 7, 0]
    valid_dup, msg_dup, _ = validate_puzzle_input(dup_input)
    print(f"  Duplicate input {dup_input} -> Valid: {valid_dup}, Message: {msg_dup}")
    assert not valid_dup and "Duplicates" in msg_dup

    oob_input = [1, 2, 3, 4, 5, 6, 7, 9, 0]
    valid_oob, msg_oob, _ = validate_puzzle_input(oob_input)
    print(f"  Out-of-range input {oob_input} -> Valid: {valid_oob}, Message: {msg_oob}")
    assert not valid_oob and "out of range" in msg_oob

    valid_input = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    valid_ok, msg_ok, parsed = validate_puzzle_input(valid_input)
    print(f"  Valid input {valid_input} -> Valid: {valid_ok}, Parsed: {parsed}")
    assert valid_ok and parsed == GOAL_STATE
    print("  >>> PASS: All validation rules enforced correctly!")

    # -------------------------------------------------------------
    # Test 5: Random solvable puzzle generator
    # -------------------------------------------------------------
    print("\n[TEST 5] Random solvable puzzle generator")
    for diff in ["Easy", "Medium", "Hard", "Random"]:
        puz = generate_random_puzzle(diff)
        inv_count = count_inversions(puz)
        solv = is_solvable(puz)
        print(f"  Difficulty: {diff:<6} -> State: {puz} | Inversions: {inv_count} | Solvable: {solv}")
        assert solv, f"Generated puzzle for {diff} was not solvable!"
        assert puz != GOAL_STATE, "Generated puzzle should not be goal state!"
    print("  >>> PASS: Solvable puzzles reliably generated with guaranteed even parity!")

    # -------------------------------------------------------------
    # Test 6: Both heuristic comparison modes & optimality
    # -------------------------------------------------------------
    print("\n[TEST 6] Both heuristic comparison modes on Medium state")
    # State requiring ~8 moves:
    test_state = (1, 2, 3, 0, 4, 6, 7, 5, 8)
    res_m_comp = solve_puzzle(test_state, manhattan_distance, "Manhattan Distance")
    res_mis_comp = solve_puzzle(test_state, misplaced_tiles, "Misplaced Tiles")

    print(f"  State: {test_state}")
    print("  " + "-" * 60)
    print(f"  {'Metric':<25} | {'Manhattan Distance':<18} | {'Misplaced Tiles':<18}")
    print("  " + "-" * 60)
    print(f"  {'Optimal Moves':<25} | {res_m_comp.moves_count:<18} | {res_mis_comp.moves_count:<18}")
    print(f"  {'Nodes Explored':<25} | {res_m_comp.nodes_explored:<18} | {res_mis_comp.nodes_explored:<18}")
    print(f"  {'Nodes Generated':<25} | {res_m_comp.nodes_generated:<18} | {res_mis_comp.nodes_generated:<18}")
    print(f"  {'Execution Time (ms)':<25} | {res_m_comp.execution_time*1000:<18.2f} | {res_mis_comp.execution_time*1000:<18.2f}")
    print(f"  {'Initial h(n)':<25} | {res_m_comp.initial_h:<18} | {res_mis_comp.initial_h:<18}")
    print("  " + "-" * 60)

    # Optimality check
    assert res_m_comp.moves_count == res_mis_comp.moves_count, "Moves must match optimal length!"
    # Dominance check: Manhattan explores fewer or equal nodes
    assert res_m_comp.nodes_explored <= res_mis_comp.nodes_explored, "Manhattan should explore <= nodes!"
    print("  >>> PASS: Both heuristics returned optimal path length! Manhattan verified dominant.")

    print("\n" + "=" * 70)
    print("ALL 6 REQUIRED TEST SCENARIOS PASSED WITH 100% SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()


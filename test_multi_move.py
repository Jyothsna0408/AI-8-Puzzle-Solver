"""
Test verification script for multi-move puzzle solution and UI integrity.
Checks:
1. Multi-move puzzle is solved correctly by both heuristics.
2. Every intermediate state from Step 0 to Step N is generated in correct order.
3. Every transition from Step (i-1) to Step (i) is physically valid on a 3x3 board.
4. Tile moved and direction match the exact blank tile displacement.
5. Goal state is reached at Step N.
6. Verification that Theory and Viva sections are NOT visible in app.py.
"""

from solver import (
    GOAL_STATE,
    manhattan_distance,
    misplaced_tiles,
    solve_puzzle,
    is_solvable,
    GRID_SIZE,
)


def verify_multi_move_solution():
    print("=" * 70)
    print("VERIFYING MULTI-MOVE PUZZLE SOLUTION & INTERMEDIATE STATES")
    print("=" * 70)

    # Multi-move test puzzles:
    puzzles = [
        ("3-move state", (1, 2, 3, 0, 4, 6, 7, 5, 8)),
        ("11-move state", (2, 5, 3, 0, 6, 8, 1, 4, 7)),
    ]

    for puz_name, multi_move_state in puzzles:
        print(f"\n==================================================")
        print(f"Testing Puzzle: {puz_name} {multi_move_state}")
        print(f"==================================================")
        assert is_solvable(multi_move_state), "State must be solvable!"

        for h_name, h_func in [
            ("Manhattan Distance", manhattan_distance),
            ("Misplaced Tiles", misplaced_tiles),
        ]:
            print(f"\n--- Testing Heuristic: {h_name} ---")
            res = solve_puzzle(multi_move_state, h_func, h_name)
            assert res.is_solved, f"Failed to solve with {h_name}"
            total_steps = len(res.path) - 1

            print(f"Optimal Moves: {res.moves_count} (Total States in path: {len(res.path)})")
            print(f"Nodes Explored: {res.nodes_explored}, Execution Time: {res.execution_time*1000:.2f} ms")

            assert res.path[0] == multi_move_state, "Step 0 must be initial state!"
            assert res.path[-1] == GOAL_STATE, f"Final step {total_steps} must be goal state!"
            assert len(res.actions) == total_steps, "Action count must equal total steps!"
            assert len(res.tiles_moved) == total_steps, "Tiles moved count must equal total steps!"
            assert len(res.directions) == total_steps, "Directions count must equal total steps!"

            # Verify EVERY intermediate state transition
            for step in range(1, total_steps + 1):
                prev_state = res.path[step - 1]
                curr_state = res.path[step]
                tile = res.tiles_moved[step - 1]
                direction = res.directions[step - 1]

                prev_blank = prev_state.index(0)
                curr_blank = curr_state.index(0)

                prev_tile_pos = prev_state.index(tile)
                curr_tile_pos = curr_state.index(tile)

                # In sliding puzzle:
                # 1. The moved tile must land exactly where the blank was
                assert curr_tile_pos == prev_blank, (
                    f"Step {step}: Tile {tile} did not land on previous blank position!"
                )
                # 2. The blank must land where the tile was
                assert curr_blank == prev_tile_pos, (
                    f"Step {step}: Blank did not land on previous tile position!"
                )

                # 3. Direction verification:
                prev_r, prev_c = prev_tile_pos // GRID_SIZE, prev_tile_pos % GRID_SIZE
                curr_r, curr_c = curr_tile_pos // GRID_SIZE, curr_tile_pos % GRID_SIZE

                dr = curr_r - prev_r
                dc = curr_c - prev_c

                dir_map = {
                    (-1, 0): "Up",
                    (1, 0): "Down",
                    (0, -1): "Left",
                    (0, 1): "Right",
                }
                expected_dir = dir_map[(dr, dc)]
                assert direction == expected_dir, (
                    f"Step {step}: Expected direction {expected_dir}, got {direction}"
                )

                print(
                    f"  Step {step}: Moved Tile {tile} {direction:<5} | "
                    f"Prev Blank: ({prev_blank//3},{prev_blank%3}) -> "
                    f"New Blank: ({curr_blank//3},{curr_blank%3})"
                )

            print(f"  >>> PASS: All {total_steps} intermediate transitions are physically exact!")

    # Verify Theory and Viva sections are NOT visible in app.py
    with open("app.py", "r", encoding="utf-8") as f:
        app_content = f.read()

    assert "Viva" not in app_content, "Theory/Viva section should not be visible in app.py!"
    assert "Theory" not in app_content, "Theory section should not be visible in app.py!"
    print("\n[VERIFICATION] Confirmed: Theory and Viva sections are NOT visible in app.py.")

    print("\n" + "=" * 70)
    print("ALL MULTI-MOVE PUZZLE TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    verify_multi_move_solution()

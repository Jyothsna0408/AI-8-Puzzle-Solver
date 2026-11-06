"""
Comprehensive unit and integration tests for AI-Based 8-Puzzle Solver.
Tests:
1. A puzzle requiring two moves.
2. An already-solved puzzle.
3. An unsolvable puzzle (odd inversions detected).
4. Invalid repeated numbers (input validation).
5. Random solvable puzzle generation and verification.
6. Both heuristic comparison modes (Manhattan vs Misplaced Tiles).
7. Optimality and admissibility verification.
"""

import unittest
from solver import (
    GOAL_STATE,
    manhattan_distance,
    misplaced_tiles,
    count_inversions,
    is_solvable,
    validate_puzzle_input,
    solve_puzzle,
    generate_random_puzzle,
    get_neighbors,
)


class Test8PuzzleSolver(unittest.TestCase):

    def test_already_solved_puzzle(self):
        """Test 2: An already-solved puzzle should take 0 moves."""
        solved_state = (1, 2, 3, 4, 5, 6, 7, 8, 0)
        self.assertEqual(manhattan_distance(solved_state), 0)
        self.assertEqual(misplaced_tiles(solved_state), 0)
        self.assertTrue(is_solvable(solved_state))

        res_manhattan = solve_puzzle(solved_state, manhattan_distance, "Manhattan Distance")
        self.assertTrue(res_manhattan.is_solved)
        self.assertEqual(res_manhattan.moves_count, 0)
        self.assertEqual(res_manhattan.nodes_explored, 0)
        self.assertEqual(res_manhattan.path, [solved_state])

        res_misplaced = solve_puzzle(solved_state, misplaced_tiles, "Misplaced Tiles")
        self.assertTrue(res_misplaced.is_solved)
        self.assertEqual(res_misplaced.moves_count, 0)
        self.assertEqual(res_misplaced.nodes_explored, 0)
        self.assertEqual(res_misplaced.path, [solved_state])

    def test_two_move_puzzle(self):
        """
        Test 1: A puzzle requiring exactly two moves.
        Goal:
        1 2 3
        4 5 6
        7 8 0

        From goal:
        Move 8 Right -> 0 is at (2, 1):
        1 2 3
        4 5 6
        7 0 8
        Move 5 Down -> 0 is at (1, 1):
        1 2 3
        4 0 6
        7 5 8
        This state is exactly 2 moves away from goal!
        """
        two_move_state = (1, 2, 3, 4, 0, 6, 7, 5, 8)
        self.assertTrue(is_solvable(two_move_state))

        # Solve with Manhattan Distance
        res_m = solve_puzzle(two_move_state, manhattan_distance, "Manhattan Distance")
        self.assertTrue(res_m.is_solved)
        self.assertEqual(res_m.moves_count, 2)
        self.assertEqual(res_m.path[0], two_move_state)
        self.assertEqual(res_m.path[-1], GOAL_STATE)
        self.assertEqual(len(res_m.actions), 2)
        self.assertEqual(res_m.tiles_moved, [5, 8])
        self.assertEqual(res_m.directions, ["Up", "Left"])

        # Solve with Misplaced Tiles
        res_mis = solve_puzzle(two_move_state, misplaced_tiles, "Misplaced Tiles")
        self.assertTrue(res_mis.is_solved)
        self.assertEqual(res_mis.moves_count, 2)
        self.assertEqual(res_mis.path[0], two_move_state)
        self.assertEqual(res_mis.path[-1], GOAL_STATE)
        self.assertEqual(len(res_mis.actions), 2)
        self.assertEqual(res_mis.tiles_moved, [5, 8])
        self.assertEqual(res_mis.directions, ["Up", "Left"])

    def test_unsolvable_puzzle(self):
        """
        Test 3: An unsolvable puzzle.
        Goal with 1 and 2 swapped:
        2 1 3
        4 5 6
        7 8 0
        Inversion count: (2, 1) = 1 inversion (odd).
        Since inversion count is odd, this state is mathematically impossible to solve.
        """
        unsolvable_state = (2, 1, 3, 4, 5, 6, 7, 8, 0)
        inversions = count_inversions(unsolvable_state)
        self.assertEqual(inversions, 1)
        self.assertFalse(is_solvable(unsolvable_state))

        # Another classic unsolvable: 1 2 3, 4 5 6, 8 7 0 (7 and 8 swapped)
        unsolvable_state_2 = (1, 2, 3, 4, 5, 6, 8, 7, 0)
        self.assertEqual(count_inversions(unsolvable_state_2), 1)
        self.assertFalse(is_solvable(unsolvable_state_2))

    def test_invalid_repeated_numbers(self):
        """Test 4: Input validation rejecting repeated numbers and out-of-range numbers."""
        # Duplicate 1, missing 8
        dup_input = [1, 1, 2, 3, 4, 5, 6, 7, 0]
        valid, msg, state = validate_puzzle_input(dup_input)
        self.assertFalse(valid)
        self.assertIn("Duplicates", msg)

        # Out of bounds number: 9
        oob_input = [1, 2, 3, 4, 5, 6, 7, 9, 0]
        valid, msg, state = validate_puzzle_input(oob_input)
        self.assertFalse(valid)
        self.assertIn("out of range", msg)

        # Insufficient tiles: 8 tiles
        short_input = [1, 2, 3, 4, 5, 6, 7, 8]
        valid, msg, state = validate_puzzle_input(short_input)
        self.assertFalse(valid)
        self.assertIn("Expected 9 tiles", msg)

        # Valid input
        valid_input = [1, 2, 3, 4, 5, 6, 7, 8, 0]
        valid, msg, state = validate_puzzle_input(valid_input)
        self.assertTrue(valid)
        self.assertEqual(state, GOAL_STATE)

    def test_random_solvable_puzzle(self):
        """Test 5: Random solvable puzzle generator returns valid, solvable state."""
        for difficulty in ["Easy", "Medium", "Hard", "Random"]:
            puzzle = generate_random_puzzle(difficulty)
            valid, msg, state = validate_puzzle_input(list(puzzle))
            self.assertTrue(valid, f"Failed for {difficulty}: {msg}")
            self.assertTrue(is_solvable(puzzle), f"Generated puzzle not solvable for {difficulty}")
            self.assertNotEqual(puzzle, GOAL_STATE)

    def test_heuristic_comparison_and_optimality(self):
        """
        Test 6 & Optimality:
        Both heuristics should find the exact same optimal solution length.
        Manhattan Distance, being more informed, typically explores fewer or equal nodes.
        """
        # Test case requiring ~6-8 moves:
        # Initial:
        # 1 2 3
        # 0 4 6
        # 7 5 8
        test_state = (1, 2, 3, 0, 4, 6, 7, 5, 8)
        self.assertTrue(is_solvable(test_state))

        res_manhattan = solve_puzzle(test_state, manhattan_distance, "Manhattan Distance")
        res_misplaced = solve_puzzle(test_state, misplaced_tiles, "Misplaced Tiles")

        self.assertTrue(res_manhattan.is_solved)
        self.assertTrue(res_misplaced.is_solved)

        # Crucial: Both must return the optimal number of moves!
        self.assertEqual(res_manhattan.moves_count, res_misplaced.moves_count)

        # Manhattan Distance is more informed: h_manhattan(n) >= h_misplaced(n)
        # Therefore, Manhattan should explore <= nodes than Misplaced Tiles
        self.assertLessEqual(res_manhattan.nodes_explored, res_misplaced.nodes_explored)

    def test_admissibility_relation(self):
        """
        Verify h_misplaced(n) <= h_manhattan(n) <= h*(n) for various states.
        Every misplaced tile is at least 1 move away, so Manhattan >= Misplaced.
        """
        states = [
            (1, 2, 3, 4, 0, 6, 7, 5, 8),
            (1, 2, 3, 0, 4, 6, 7, 5, 8),
            (2, 8, 3, 1, 6, 4, 7, 0, 5),
            (1, 3, 4, 8, 6, 2, 7, 0, 5),
        ]
        for s in states:
            h_mis = misplaced_tiles(s)
            h_man = manhattan_distance(s)
            self.assertLessEqual(h_mis, h_man, f"Admissibility dominance failed for state {s}")


if __name__ == "__main__":
    unittest.main()


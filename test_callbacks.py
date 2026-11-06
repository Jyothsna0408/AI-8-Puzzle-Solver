"""
Direct unit test for app.py callbacks and state management.
Verifies:
1. on_generate_puzzle updates all 9 input cells in session state.
2. Easy, Medium, Hard difficulty levels work properly.
3. Generated puzzles have all 9 numbers 0-8 exactly once and are solvable.
4. on_reset_default restores DEFAULT_STATE (1, 2, 3, 4, 5, 6, 7, 0, 8).
5. Previous solution results and current_step are cleared by both callbacks.
"""

import streamlit as st
from solver import (
    GOAL_STATE,
    DEFAULT_STATE,
    is_solvable,
    count_inversions,
    validate_puzzle_input,
)
from app import (
    on_generate_puzzle,
    on_reset_default,
    DEFAULT_INITIAL_PUZZLE,
)


def run_callback_tests():
    print("=" * 70)
    print("TESTING APP.PY CALLBACKS & STATE MANAGEMENT")
    print("=" * 70)

    # 1. Initialize session state
    for i in range(9):
        st.session_state[f"tile_in_{i}"] = DEFAULT_INITIAL_PUZZLE[i]
    st.session_state.solution_result = "some_previous_result"
    st.session_state.current_step = 5
    st.session_state.feedback_msg = None

    # Check default initial values
    initial_vals = [st.session_state[f"tile_in_{i}"] for i in range(9)]
    print(f"Initial State: {initial_vals}")
    assert initial_vals == [1, 2, 3, 4, 5, 6, 7, 0, 8], f"Wrong default: {initial_vals}"
    print(">>> PASS: Initial state is correctly [1, 2, 3, 4, 5, 6, 7, 0, 8]!")

    # 2. Test on_generate_puzzle for Easy, Medium, Hard
    for diff in ["Easy (2-5 moves)", "Medium (8-14 moves)", "Hard (15-25 moves)"]:
        st.session_state["difficulty_selector"] = diff
        st.session_state.solution_result = "previous_result"
        st.session_state.current_step = 3

        # Call the callback
        on_generate_puzzle()

        gen_vals = [st.session_state[f"tile_in_{i}"] for i in range(9)]
        print(f"\nGenerated for {diff}: {gen_vals}")

        # Assertions
        assert sorted(gen_vals) == list(range(9)), f"Not a valid permutation of 0-8: {gen_vals}"
        assert is_solvable(tuple(gen_vals)), f"Generated puzzle not solvable: {gen_vals}"
        assert tuple(gen_vals) != GOAL_STATE, "Generated puzzle must not be goal state!"
        assert st.session_state.solution_result is None, "Solution result not cleared!"
        assert st.session_state.current_step == 0, "Current step not reset to 0!"
        assert st.session_state.feedback_msg is not None, "Feedback message not set!"
        assert st.session_state.feedback_msg["type"] == "success"
        print(f"  Feedback: {st.session_state.feedback_msg['text']}")
        print(f"  Inversions: {count_inversions(tuple(gen_vals))} (Even Parity)")
        print(f"  >>> PASS: {diff} generation verified!")

    # 3. Test on_reset_default
    print("\n--- Testing on_reset_default ---")
    st.session_state.solution_result = "previous_result"
    st.session_state.current_step = 8

    on_reset_default()

    reset_vals = [st.session_state[f"tile_in_{i}"] for i in range(9)]
    print(f"After Reset to Default: {reset_vals}")
    assert reset_vals == [1, 2, 3, 4, 5, 6, 7, 0, 8], f"Reset values incorrect: {reset_vals}"
    assert st.session_state.solution_result is None, "Solution result not cleared!"
    assert st.session_state.current_step == 0, "Current step not reset to 0!"
    assert st.session_state.feedback_msg is not None, "Feedback message not set!"
    assert st.session_state.feedback_msg["type"] == "info"
    print(f"  Feedback: {st.session_state.feedback_msg['text']}")
    print("  >>> PASS: Reset to Default verified!")

    # 4. Verify repeated alternate clicks
    print("\n--- Testing 10 Repeated Alternate Invocations ---")
    for cycle in range(1, 11):
        on_generate_puzzle()
        gen = [st.session_state[f"tile_in_{i}"] for i in range(9)]
        assert sorted(gen) == list(range(9))
        assert is_solvable(tuple(gen))

        on_reset_default()
        res = [st.session_state[f"tile_in_{i}"] for i in range(9)]
        assert res == [1, 2, 3, 4, 5, 6, 7, 0, 8]

    print(">>> PASS: 10 consecutive Generate -> Reset cycles succeeded with 100% consistency!")

    print("\n" + "=" * 70)
    print("ALL CALLBACK TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    run_callback_tests()


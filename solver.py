"""
AI-Based 8-Puzzle Game Solver
Algorithms: A* Search with Manhattan Distance and Misplaced Tiles Heuristics.

This module provides:
- PuzzleState representation and neighbor generation.
- Heuristic functions: Manhattan Distance and Misplaced Tiles.
- Inversion-based solvability validation.
- A* Search engine with comprehensive performance metrics.
- Random solvable puzzle generator.
"""

from __future__ import annotations
import heapq
import time
import random
from typing import List, Tuple, Dict, Optional, Callable, Any

# Standard 8-puzzle Goal State:
# 1 2 3
# 4 5 6
# 7 8 0
GOAL_STATE: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 0)

# Default initial puzzle (1 move from goal):
# 1 2 3
# 4 5 6
# 7 0 8
DEFAULT_STATE: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 0, 8)
GRID_SIZE = 3

# Precomputed target coordinates for each tile 1..8
# (Tile 0 is the blank and not counted in heuristics)
GOAL_POSITIONS: Dict[int, Tuple[int, int]] = {
    tile: ((tile - 1) // GRID_SIZE, (tile - 1) % GRID_SIZE)
    for tile in range(1, 9)
}
GOAL_POSITIONS[0] = (2, 2)  # Position of blank in goal


class PuzzleNode:
    """Represents a node in the A* search tree."""

    __slots__ = ("state", "parent", "action", "tile_moved", "direction", "g", "h", "f")

    def __init__(
        self,
        state: Tuple[int, ...],
        parent: Optional[PuzzleNode] = None,
        action: Optional[str] = None,
        tile_moved: Optional[int] = None,
        direction: Optional[str] = None,
        g: int = 0,
        h: int = 0,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.tile_moved = tile_moved
        self.direction = direction
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other: PuzzleNode) -> bool:
        # Tie-breaker when f-scores are equal: prefer lower h (closer to goal)
        if self.f == other.f:
            return self.h < other.h
        return self.f < other.f


# ---------------------------------------------------------
# Heuristic Functions
# ---------------------------------------------------------

def manhattan_distance(state: Tuple[int, ...]) -> int:
    """
    Computes Manhattan Distance heuristic:
    Sum of horizontal and vertical distances of each numbered tile (1-8)
    from its target position in the goal state.
    Blank space (0) is explicitly excluded.
    """
    distance = 0
    for idx, tile in enumerate(state):
        if tile != 0:
            current_row = idx // GRID_SIZE
            current_col = idx % GRID_SIZE
            target_row, target_col = GOAL_POSITIONS[tile]
            distance += abs(current_row - target_row) + abs(current_col - target_col)
    return distance


def misplaced_tiles(state: Tuple[int, ...]) -> int:
    """
    Computes Misplaced Tiles heuristic:
    Count of numbered tiles (1-8) that are not in their goal position.
    Blank space (0) is explicitly excluded.
    """
    count = 0
    for idx, tile in enumerate(state):
        if tile != 0:
            if tile != GOAL_STATE[idx]:
                count += 1
    return count


# ---------------------------------------------------------
# Solvability & Validation Functions
# ---------------------------------------------------------

def count_inversions(state: Tuple[int, ...]) -> int:
    """
    Count the number of inversions in the flattened state, excluding 0.
    An inversion is any pair (a, b) where a > b and a appears before b.
    """
    tiles = [tile for tile in state if tile != 0]
    inversions = 0
    num_tiles = len(tiles)
    for i in range(num_tiles):
        for j in range(i + 1, num_tiles):
            if tiles[i] > tiles[j]:
                inversions += 1
    return inversions


def is_solvable(state: Tuple[int, ...]) -> bool:
    """
    For a 3x3 8-puzzle with odd width (N=3), moving tiles preserves
    the parity of inversion count. The goal state (1,2,3,4,5,6,7,8,0)
    has 0 inversions (even).
    Therefore, any configuration is solvable if and only if
    its inversion count is even.
    """
    return count_inversions(state) % 2 == 0


def validate_puzzle_input(values: List[Any]) -> Tuple[bool, str, Optional[Tuple[int, ...]]]:
    """
    Validates user input:
    - Must have exactly 9 elements.
    - Each must be an integer between 0 and 8.
    - Every number from 0 to 8 must appear exactly once.
    Returns: (is_valid, error_message, tuple_state)
    """
    if len(values) != 9:
        return False, f"Expected 9 tiles, but got {len(values)}.", None

    parsed: List[int] = []
    for i, v in enumerate(values):
        try:
            num = int(v)
        except (ValueError, TypeError):
            return False, f"Tile at position {i+1} is not a valid integer.", None

        if num < 0 or num > 8:
            return False, f"Tile value {num} is out of range [0, 8].", None
        parsed.append(num)

    unique_nums = set(parsed)
    if len(unique_nums) != 9:
        missing = sorted(list(set(range(9)) - unique_nums))
        duplicates = [x for x in unique_nums if parsed.count(x) > 1]
        return False, f"Each number from 0 to 8 must appear exactly once. Missing: {missing}, Duplicates: {duplicates}", None

    state = tuple(parsed)
    return True, "", state


# ---------------------------------------------------------
# Successor State Generation
# ---------------------------------------------------------

# Valid blank moves: (row_delta, col_delta, direction_name)
# Note: "Up" means blank moves Up, which causes the tile above it to move Down.
DIRECTIONS = [
    (-1, 0, "Up"),
    (1, 0, "Down"),
    (0, -1, "Left"),
    (0, 1, "Right"),
]


def get_neighbors(state: Tuple[int, ...]) -> List[Tuple[Tuple[int, ...], str, int, str]]:
    """
    Finds all valid successor states by sliding tiles into the blank space (0).
    Returns list of: (new_state, action_description, tile_moved, direction)
    """
    blank_idx = state.index(0)
    blank_r = blank_idx // GRID_SIZE
    blank_c = blank_idx % GRID_SIZE

    neighbors: List[Tuple[Tuple[int, ...], str, int, str]] = []
    state_list = list(state)

    for dr, dc, direction in DIRECTIONS:
        nr, nc = blank_r + dr, blank_c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            target_idx = nr * GRID_SIZE + nc
            tile_moved = state_list[target_idx]

            # Swap blank with the adjacent tile
            next_state_list = state_list.copy()
            next_state_list[blank_idx], next_state_list[target_idx] = (
                next_state_list[target_idx],
                next_state_list[blank_idx],
            )

            # Descriptive move: e.g. "Move 4 Down" or "Blank moved Up"
            opposite_dir = {
                "Up": "Down",
                "Down": "Up",
                "Left": "Right",
                "Right": "Left",
            }[direction]
            action_desc = f"Move tile {tile_moved} {opposite_dir}"

            neighbors.append((tuple(next_state_list), action_desc, tile_moved, opposite_dir))

    return neighbors


# ---------------------------------------------------------
# A* Search Engine
# ---------------------------------------------------------

class SearchResult:
    """Stores the complete outcome and performance metrics of an A* search."""

    def __init__(
        self,
        heuristic_name: str,
        is_solved: bool,
        path: List[Tuple[int, ...]],
        actions: List[str],
        moves_count: int,
        nodes_explored: int,
        nodes_generated: int,
        execution_time: float,
        initial_h: int,
        tiles_moved: Optional[List[int]] = None,
        directions: Optional[List[str]] = None,
    ):
        self.heuristic_name = heuristic_name
        self.is_solved = is_solved
        self.path = path
        self.actions = actions
        self.moves_count = moves_count
        self.nodes_explored = nodes_explored
        self.nodes_generated = nodes_generated
        self.execution_time = execution_time
        self.initial_h = initial_h
        self.tiles_moved = tiles_moved or []
        self.directions = directions or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Heuristic": self.heuristic_name,
            "Solved": self.is_solved,
            "Moves": self.moves_count,
            "Nodes Explored": self.nodes_explored,
            "Nodes Generated": self.nodes_generated,
            "Execution Time (s)": round(self.execution_time, 4),
            "Initial Heuristic Value": self.initial_h,
        }


def solve_puzzle(
    initial_state: Tuple[int, ...],
    heuristic_func: Callable[[Tuple[int, ...]], int],
    heuristic_name: str = "Heuristic",
    max_nodes: int = 200000,
) -> SearchResult:
    """
    Executes A* Search on the 8-puzzle using the given heuristic function.
    Formula: f(n) = g(n) + h(n)

    Both Manhattan Distance and Misplaced Tiles use this exact same search implementation.
    """
    start_time = time.perf_counter()

    initial_h = heuristic_func(initial_state)

    # Edge case: already solved
    if initial_state == GOAL_STATE:
        duration = time.perf_counter() - start_time
        return SearchResult(
            heuristic_name=heuristic_name,
            is_solved=True,
            path=[initial_state],
            actions=[],
            moves_count=0,
            nodes_explored=0,
            nodes_generated=1,
            execution_time=duration,
            initial_h=initial_h,
        )

    # Priority queue: entries are (f, h, counter, node)
    # Using unique counter ensures stability and eliminates object comparison issues
    open_heap: List[Tuple[int, int, int, PuzzleNode]] = []
    counter = 0

    start_node = PuzzleNode(
        state=initial_state,
        parent=None,
        action=None,
        tile_moved=None,
        g=0,
        h=initial_h,
    )
    heapq.heappush(open_heap, (start_node.f, start_node.h, counter, start_node))
    counter += 1

    # Best g score discovered for each state
    g_score: Dict[Tuple[int, ...], int] = {initial_state: 0}
    # Closed set for states whose optimal paths have been finalized
    closed_set = set()

    nodes_explored = 0
    nodes_generated = 1

    while open_heap:
        _, _, _, current_node = heapq.heappop(open_heap)
        current_state = current_node.state

        if current_state in closed_set:
            continue

        closed_set.add(current_state)
        nodes_explored += 1

        # Goal check
        if current_state == GOAL_STATE:
            duration = time.perf_counter() - start_time
            # Reconstruct solution path
            path: List[Tuple[int, ...]] = []
            actions: List[str] = []
            tiles_moved: List[int] = []
            directions: List[str] = []
            curr: Optional[PuzzleNode] = current_node
            while curr is not None:
                path.append(curr.state)
                if curr.action is not None:
                    actions.append(curr.action)
                if curr.tile_moved is not None:
                    tiles_moved.append(curr.tile_moved)
                if curr.direction is not None:
                    directions.append(curr.direction)
                curr = curr.parent

            path.reverse()
            actions.reverse()
            tiles_moved.reverse()
            directions.reverse()

            return SearchResult(
                heuristic_name=heuristic_name,
                is_solved=True,
                path=path,
                actions=actions,
                moves_count=len(actions),
                nodes_explored=nodes_explored,
                nodes_generated=nodes_generated,
                execution_time=duration,
                initial_h=initial_h,
                tiles_moved=tiles_moved,
                directions=directions,
            )

        if nodes_explored >= max_nodes:
            break

        # Generate neighbor states
        for neighbor_state, action_desc, tile_moved, move_dir in get_neighbors(current_state):
            nodes_generated += 1
            tentative_g = current_node.g + 1

            if neighbor_state in closed_set:
                continue

            if neighbor_state not in g_score or tentative_g < g_score[neighbor_state]:
                g_score[neighbor_state] = tentative_g
                h_val = heuristic_func(neighbor_state)
                child_node = PuzzleNode(
                    state=neighbor_state,
                    parent=current_node,
                    action=action_desc,
                    tile_moved=tile_moved,
                    direction=move_dir,
                    g=tentative_g,
                    h=h_val,
                )
                heapq.heappush(
                    open_heap, (child_node.f, child_node.h, counter, child_node)
                )
                counter += 1

    duration = time.perf_counter() - start_time
    return SearchResult(
        heuristic_name=heuristic_name,
        is_solved=False,
        path=[],
        actions=[],
        moves_count=0,
        nodes_explored=nodes_explored,
        nodes_generated=nodes_generated,
        execution_time=duration,
        initial_h=initial_h,
    )


# ---------------------------------------------------------
# Random Solvable Puzzle Generator
# ---------------------------------------------------------

def generate_random_puzzle(difficulty: str = "Medium") -> Tuple[int, ...]:
    """
    Generates a guaranteed valid, solvable 8-puzzle.
    difficulty:
      - 'Easy': approximately 2–5 moves from goal
      - 'Medium': approximately 8–14 moves from goal
      - 'Hard': approximately 15–25 moves from goal
      - 'Random': general solvable configuration
    Generates puzzles by making valid random moves from the goal state,
    avoiding immediate reversals.
    """
    steps_map = {
        "Easy": (2, 5),
        "Medium": (8, 14),
        "Hard": (15, 25),
    }

    min_steps, max_steps = steps_map.get(difficulty, (8, 14))

    for _ in range(100):
        steps = random.randint(min_steps, max_steps)
        current = GOAL_STATE
        last_state = None
        for _ in range(steps):
            nbrs = [n[0] for n in get_neighbors(current) if n[0] != last_state]
            if not nbrs:
                nbrs = [n[0] for n in get_neighbors(current)]
            last_state = current
            current = random.choice(nbrs)

        if current != GOAL_STATE and is_solvable(current) and len(set(current)) == 9:
            return current

    # Fallback to random permutation with even inversions
    while True:
        nums = list(range(9))
        random.shuffle(nums)
        cand = tuple(nums)
        if cand != GOAL_STATE and is_solvable(cand):
            return cand


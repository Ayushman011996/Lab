import numpy as np
from queue import PriorityQueue

# Class to represent a state of the puzzle
class State:
    def __init__(self, state, parent):
        self.state = state  # Current state of the puzzle
        self.parent = parent  # Parent state (for backtracking the solution path)

    def __lt__(self, other):
        return False  # Required for PriorityQueue but not used in comparison

# Class to represent the Puzzle and its solving logic
class Puzzle:
    def __init__(self, initial_state, goal_state):
        self.initial_state = initial_state  # Initial puzzle state
        self.goal_state = goal_state  # Goal state of the puzzle

    def print_state(self, state):
        # Print the current state of the puzzle
        print(state)

    def is_goal(self, state):
        # Check if the current state matches the goal state
        return np.array_equal(state, self.goal_state)

    def get_possible_moves(self, state):
        # Generate all possible moves from the current state
        possible_moves = []
        zero_pos = np.argwhere(state == 0)[0]  # Find the position of the empty tile (0)
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Left, Right, Up, Down

        for dx, dy in directions:
            new_pos = (zero_pos[0] + dx, zero_pos[1] + dy)  # Calculate new position

            if 0 <= new_pos[0] < 3 and 0 <= new_pos[1] < 3:  # Check boundaries
                new_state = np.copy(state)  # Copy the current state
                # Swap the empty tile with the target tile
                new_state[zero_pos[0]][zero_pos[1]], new_state[new_pos[0]][ new_pos[1]] = (
                    new_state[new_pos[0]][ new_pos[1]],
                    new_state[zero_pos[0]][ zero_pos[1]],
                )
                possible_moves.append(new_state)  # Add the new state to possible moves

        return possible_moves

    def heuristic(self, state):
        # Calculate the heuristic: number of misplaced tiles
        return np.sum(state != self.goal_state)

    def solve(self):
        # Solve the puzzle using A* algorithm
        queue = PriorityQueue()
        queue.put((0, State(self.initial_state, None)))  # Initial state with priority 0
        visited = set()  # Keep track of visited states

        while not queue.empty():
            _, current_state = queue.get()  # Get the state with the highest priority

            if self.is_goal(current_state.state):
                return current_state  # Return the goal state

            # Explore all possible moves
            for move in self.get_possible_moves(current_state.state):
                move_tuple = tuple(map(tuple, move))  # Convert state to tuple for hashing

                if move_tuple not in visited:  # Check if the state is not visited
                    visited.add(move_tuple)  # Mark the state as visited
                    priority = self.heuristic(move)  # Calculate priority based on heuristic
                    queue.put((priority, State(move, current_state)))  # Add to queue

        return None  # Return None if no solution is found

# Test the Puzzle Solver
initial_state = np.array([[2, 8, 1], [0, 4, 3], [7, 6, 5]])  # Initial puzzle configuration
goal_state = np.array([[1, 2, 3], [8, 0, 4], [7, 6, 5]])  # Goal puzzle configuration

puzzle = Puzzle(initial_state, goal_state)
solution = puzzle.solve()

if solution:
    moves = []
    while solution:
        moves.append(solution.state)  # Trace back the solution path
        solution = solution.parent

    move_count = len(moves) - 1
    for move in reversed(moves):  # Print the solution moves in order
        puzzle.print_state(move)
        print("-")
    print(f"Number of moves: {move_count}")
else:
    print("No solution found.")

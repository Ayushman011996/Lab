import numpy as np

def hill_climbing(func, start, step_size=0.01, max_iterations=1000):
  
    current_position = start
    current_value = func(current_position)

    for _ in range(max_iterations):
        # Calculate values at neighboring points
        positive_neighbor = current_position + step_size
        negative_neighbor = current_position - step_size
        
        positive_value = func(positive_neighbor)
        negative_value = func(negative_neighbor)

        # Move to the better neighbor if it improves the current value
        if positive_value > current_value and positive_value >= negative_value:
            current_position = positive_neighbor
            current_value = positive_value
        elif negative_value > current_value and negative_value > positive_value:
            current_position = negative_neighbor
            current_value = negative_value
        else:
            # Stop if no improvement is found
            break

    return current_position, current_value

# Input the function to maximize from the user
while True:
    try:
        func_str = input("Enter a function of x (e.g., -x**2 + 4*x): ")
        x = 0  # Test the function with a dummy value
        eval(func_str)
        break
    except Exception as e:
        print(f"Invalid function. Please try again. Error: {e}")

# Convert the input string into a lambda function
func = lambda x: eval(func_str)

# Input the starting point for the search
while True:
    try:
        start = float(input("Enter the starting value to begin the search: "))
        break
    except ValueError:
        print("Invalid input. Please enter a numeric value.")

# Perform hill climbing and display the results
maxima, max_value = hill_climbing(func, start)
print(f"The maxima is at x = {maxima}")
print(f"The maximum value obtained is {max_value}")

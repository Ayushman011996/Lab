def minimax(tree, index, is_max, alpha, beta):
    if index >= len(tree) // 2:  # If it's a leaf node, return the value
        return tree[index]

    if is_max:
        best = float('-inf')
        for child in [index * 2 + 1, index * 2 + 2]:  # Left & Right child
            if child < len(tree):
                best = max(best, minimax(tree, child, False, alpha, beta))
                alpha = max(alpha, best)
                if beta <= alpha:  # Prune
                    break
        return best
    else:
        best = float('inf')
        for child in [index * 2 + 1, index * 2 + 2]:  # Left & Right child
            if child < len(tree):
                best = min(best, minimax(tree, child, True, alpha, beta))
                beta = min(beta, best)
                if beta <= alpha:  # Prune
                    break
        return best

# Example input
tree = [0, 0, 0, 0, 0, 0, 0, 2, 3, 5, 9, 0, 1, 7, 5]
result = minimax(tree, 0, True, float('-inf'), float('inf'))
print("Evaluated Value:", result)

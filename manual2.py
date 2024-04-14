rates = {
    'SeaShells': {'Pizza Slice': 1.41, 'Wasabi Root': 0.61, 'Snowball': 2.08, 'SeaShells': 1},
    'Pizza Slice': {'SeaShells': 0.71, 'Wasabi Root': 0.48, 'Snowball': 1.52, 'Pizza Slice': 1},
    'Wasabi Root': {'SeaShells': 1.56, 'Pizza Slice': 2.05, 'Snowball': 3.26, 'Wasabi Root': 1},
    'Snowball': {'SeaShells': 0.46, 'Pizza Slice': 0.64, 'Wasabi Root': 0.3, 'Snowball': 1}
}

def calculate_seashells_return(starting_amount, path):
    amount = starting_amount
    amounts = []
    for i in range(len(path) - 1):
        amounts.append(rates[path[i]][path[i+1]])
        amount *= rates[path[i]][path[i+1]]
    print(amounts)
    return amount, amounts

currencies = ['Pizza Slice', 'Wasabi Root', 'Snowball', 'SeaShells']
paths = []

for i in currencies:
    for j in currencies:
        for k in currencies:
            for l in currencies:
                paths.append(['SeaShells', i, j, k, l, 'SeaShells'])

# Calculate the return for each path
results = []
for path in paths:
    seashells_return, amts = calculate_seashells_return(1, path)
    results.append((path, seashells_return, amts))

# Find the path with the maximum return
max_value = float('-inf')
max_return_path = None
for val in results:
    if val[1] > max_value:
        max_value = val[1]
        max_return_path = val

print(max_return_path)

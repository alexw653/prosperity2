import random
import argparse

def pick_number():
    # Set the range of numbers
    min_number = 900
    max_number = 1000
    
    # Calculate the total probability area under the linear distribution
    total_probability = sum(range(1, max_number - min_number + 2))  # Sum of 1 to (max_number - min_number + 1)
    
    # Generate a random value within the total probability area
    rand_val = random.randint(1, total_probability)
    
    # Determine which number corresponds to the selected random value
    current_sum = 0
    for number in range(min_number, max_number + 1):
        current_sum += number - min_number + 1
        if rand_val <= current_sum:
            return number

# Simulate picking a number multiple times
num_simulations = 10000
results = {number: 0 for number in range(900, 1001)}

def main():
    trials = 10000
    nums = [pick_number() for _ in range(trials)]

    profits = []
    for i in range(900, 1001):
        for j in range(i, 1001):
            profit = 0
            for num in nums:
                if num <= i:
                    profit += 1000 - i
                elif num <= j:
                    profit += 1000 - j
            profits.append((profit, i, j))
    print(sorted(profits)[::-1][:3])

if __name__ == "__main__":
    main()

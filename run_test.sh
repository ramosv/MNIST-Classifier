#!/bin/bash

# Shell Script to Run main.py with Different Arguments and Capture Final Test Outputs

# Initialize or clear the results.txt file
echo "=== CNN MNIST Classification Results ===" > results.txt
echo "" >> results.txt

# Make an array of test configurations
# Each configuration is an argument to pass to main.py
declare -a tests=(
    "--epochs 5 --lr 0.01"
    "--epochs 5 --lr 0.001"
    "--epochs 5 --lr 0.005"
    "--epochs 5 --lr 0.0005"
    "--epochs 5 --lr 0.002"
)

# Loop through each test configuration
for i in "${!tests[@]}"; do
    test_num=$((i+1))
    config=${tests[$i]}
    
    echo "Running Test $test_num with configuration: $config"
    echo "Test $test_num:" >> results.txt
    echo "Configuration: $config" >> results.txt
    
    # Run the Python script with configuration
    # Get the output
    output=$(python main.py $config | grep "Test set:")
    
    # Add the output to results.txt
    echo "$output" >> results.txt
    echo "" >> results.txt

    echo "Test $test_num completed. Result appended to results.txt."
    echo "--------------------------------------------"
done

# Train CNN with Half and 5% of Training Data
echo "=== Task 3: Training with Reduced Data Sizes ===" >> results.txt
echo "" >> results.txt

declare -a task3_tests=(
    "--epochs 5 --lr 1.0 --data-fraction 0.5"
    "--epochs 5 --lr 1.0 --data-fraction 0.05"
)

for i in "${!task3_tests[@]}"; do
    test_num=$((i+1))
    config=${task3_tests[$i]}
    
    echo "Running Task 3 Test $test_num with configuration: $config"
    echo "Task 3 Test $test_num:" >> results.txt
    echo "Configuration: $config" >> results.txt
    
    # Run the Python script with the specified configuration
    # Get the output
    output=$(python main.py $config | grep "Test set:")
    
    # Add the output to results.txt
    echo "$output" >> results.txt
    echo "" >> results.txt

    echo "Task 3 Test $test_num completed. Result appended to results.txt."
    echo "--------------------------------------------"
done

# Adding Noise to Test Images
echo "=== Task 4: Adding Noise to Test Images ===" >> results.txt
echo "" >> results.txt

# Setup arrar with noise parameters for Gaussian and Salt & Pepper noise
declare -a noise_tests=(
    "gaussian 0 0.1"
    "gaussian 0 0.3"
    "gaussian 0 0.5"
    "gaussian 0 0.7"
    "s&p 0.05"
    "s&p 0.1"
    "s&p 0.2"
    "s&p 0.3"
)

for i in "${!noise_tests[@]}"; do
    test_num=$((i+1))
    noise_type=${noise_tests[$i]}
    
    echo "Running Task 4 Noise Test $test_num with noise type: $noise_type"
    echo "Task 4 Noise Test $test_num:" >> results.txt
    echo "Noise Type: $noise_type" >> results.txt
    
    # Execute the noise addition and testing using a separate Python script
    # Pass the noise parameters as arguments
    output=$(python add_noise_test.py --noise-type $noise_type)
    
    # Add the output to results.txt
    echo "$output" >> results.txt
    echo "" >> results.txt

    echo "Task 4 Noise Test $test_num completed. Result appended to results.txt."
    echo "--------------------------------------------"
done

echo "All tests completed. Check results.txt for the final outputs."
from CNN import Run_Model
from Noise import Add_Noise

def main():
    # Initialize or clear the results.txt file
    with open('results.txt', 'w') as f:
        f.write("=== MNIST Classification Results ===\n\n")
    
    # Task 1 and 2: Array of test configurations for CNN and FNN
    tests = [
        {'model_type': 'CNN', 'epochs': 5, 'lr': 0.01},
        {'model_type': 'CNN', 'epochs': 5, 'lr': 0.001},
        {'model_type': 'FNN', 'epochs': 5, 'lr': 0.005},
        {'model_type': 'FNN', 'epochs': 5, 'lr': 0.0005},
        {'model_type': 'CNN', 'epochs': 5, 'lr': 0.002}
    ]

    for i, config in enumerate(tests):
        test_num = i + 1
        print(f"Running Test {test_num} with configuration: {config}")
        with open('results.txt', 'a') as f:
            f.write(f"Test {test_num}:\n")
            f.write(f"Configuration: {config}\n")

        # Running the specified model (CNN or FNN) with the configuration
        output_lines = Run_Model(model_type=config['model_type'], epochs=config['epochs'], lr=config['lr'])

        # Filtering the output to only include lines containing "Test set:"
        test_output = [line for line in output_lines if "Test set:" in line]
        test_output_str = "\n".join(test_output)

        with open('results.txt', 'a') as f:
            f.write(f"{test_output_str}\n\n")

        print(f"Test {test_num} completed. Result appended to results.txt.")
        print("--------------------------------------------")

    # Task 3: Training with Reduced Data Sizes (only CNN for simplicity)
    with open('results.txt', 'a') as f:
        f.write("=== Task 3: Training with Reduced Data Sizes ===\n\n")

    task3_tests = [
        {'model_type': 'CNN', 'epochs': 5, 'lr': 1.0, 'percent': 0.5},
        {'model_type': 'CNN', 'epochs': 5, 'lr': 1.0, 'percent': 0.05}
    ]

    for i, config in enumerate(task3_tests):
        test_num = i + 1
        print(f"Running Task 3 Test {test_num} with configuration: {config}")
        with open('results.txt', 'a') as f:
            f.write(f"Task 3 Test {test_num}:\n")
            f.write(f"Configuration: {config}\n")

        # Running the CNN with reduced data sizes
        output_lines = Run_Model(model_type=config['model_type'], epochs=config['epochs'], lr=config['lr'], percent=config['percent'])

        # Filtering the output to only include lines containing "Test set:"
        test_output = [line for line in output_lines if "Test set:" in line]
        test_output_str = "\n".join(test_output)

        with open('results.txt', 'a') as f:
            f.write(f"{test_output_str}\n\n")

        print(f"Task 3 Test {test_num} completed. Result appended to results.txt.")
        print("--------------------------------------------")

    # Task 4: Adding Noise to Test Images
    with open('results.txt', 'a') as f:
        f.write("=== Task 4: Adding Noise to Test Images ===\n\n")

    noise_tests = [
        {'noise_type': 'gaussian', 'mean': 0, 'std': 0.25},
        {'noise_type': 'gaussian', 'mean': 0, 'std': 0.50},
        {'noise_type': 'gaussian', 'mean': 0, 'std': 0.75},
        {'noise_type': 'gaussian', 'mean': 0, 'std': 1.0},
        {'noise_type': 's&p', 'prob': 0.25},
        {'noise_type': 's&p', 'prob': 0.50},
        {'noise_type': 's&p', 'prob': 0.75},
        {'noise_type': 's&p', 'prob': 1.0}
    ]

    for i, noise_config in enumerate(noise_tests):
        test_num = i + 1
        print(f"Running Task 4 Noise Test {test_num} with noise type: {noise_config}")
        with open('results.txt', 'a') as f:
            f.write(f"Task 4 Noise Test {test_num}:\n")
            f.write(f"Noise Type: {noise_config}\n")

        # Run the noise addition and testing
        output_lines = Add_Noise(**noise_config)

        # Collect outputs
        test_output_str = "\n".join(output_lines)

        with open('results.txt', 'a') as f:
            f.write(f"{test_output_str}\n\n")

        print(f"Task 4 Noise Test {test_num} completed. Result appended to results.txt.")
        print("--------------------------------------------")

    print("All tests completed. Check results.txt for the final outputs.")

if __name__ == '__main__':
    main()
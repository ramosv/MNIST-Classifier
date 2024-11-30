import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os

# Import the NeuralNet class from main.py
from main import NeuralNet

def add_gaussian_noise(image, mean=0.0, std=0.5):
    noisy_img = image + torch.randn(image.size()) * std + mean
    noisy_img = torch.clamp(noisy_img, 0., 1.)
    return noisy_img

def add_salt_pepper_noise(image, prob=0.2):
    noisy_img = image.clone()
    num_pixels = image.numel()
    num_salt = int(prob * num_pixels / 2)
    num_pepper = int(prob * num_pixels / 2)

    # Salt noise
    coords = np.random.choice(num_pixels, num_salt, replace=False)
    noisy_img.view(-1)[coords] = 1.0

    # Pepper noise
    coords = np.random.choice(num_pixels, num_pepper, replace=False)
    noisy_img.view(-1)[coords] = 0.0

    return noisy_img

def main():
    parser = argparse.ArgumentParser(description='Add Noise to Test Images and Evaluate CNN')
    parser.add_argument('--noise-type', type=str, required=True,
                        help='Type of noise to add: "gaussian" or "s&p"')
    parser.add_argument('--mean', type=float, default=0.0, help='Mean for Gaussian noise')
    parser.add_argument('--std', type=float, default=0.5, help='Standard deviation for Gaussian noise')
    parser.add_argument('--prob', type=float, default=0.2, help='Probability for Salt & Pepper noise')
    parser.add_argument('--save-images', action='store_true', default=False,
                        help='Save the noisy images to disk')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = NeuralNet().to(device)
    if not os.path.exists("mnist_cnn.pt"):
        print("Trained model 'mnist_cnn.pt' not found. Please train the model first.")
        return
    model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
    model.eval()

    # Load the test dataset
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)

    # Identify two correctly classified test images
    correct_images = []
    correct_labels = []
    with torch.no_grad():
        for data, target in test_dataset:
            data, target = data.to(device), target.to(device)
            output = model(data.unsqueeze(0))
            pred = output.argmax(dim=1, keepdim=True)
            if pred.item() == target.item():
                correct_images.append(data.cpu())
                correct_labels.append(target.item())
                if len(correct_images) == 2:
                    break

    if len(correct_images) < 2:
        print("Not enough correctly classified images found.")
        return

    results = ""

    for idx in range(2):
        image = correct_images[idx]
        label = correct_labels[idx]
        if args.noise_type.lower() == "gaussian":
            noisy_image = add_gaussian_noise(image, mean=args.mean, std=args.std)
            noise_param = f"Gaussian Noise (mean={args.mean}, std={args.std})"
        elif args.noise_type.lower() in ["s&p", "salt_pepper", "salt and pepper"]:
            noisy_image = add_salt_pepper_noise(image, prob=args.prob)
            noise_param = f"Salt and Pepper Noise (prob={args.prob})"
        else:
            print("Unsupported noise type. Use 'gaussian' or 's&p'.")
            return

        # Predict with noisy image
        output = model(noisy_image.unsqueeze(0).to(device))
        pred = output.argmax(dim=1, keepdim=True).item()

        # Check if tricked
        fooled = pred != label
        result = f"Image {idx+1} - Original Label: {label}, Predicted: {label}, After Noise: {pred} | Fooled: {fooled}"
        results += result + "\n"

        # Save the noisy images
        if args.save_images:
            import matplotlib.pyplot as plt
            plt.imshow(noisy_image.squeeze(), cmap='gray')
            plt.title(f'Original: {label}, Predicted: {pred}')
            plt.savefig(f'noisy_image_{idx+1}.png')
            plt.close()

    print(results)

if __name__ == "__main__":
    main()
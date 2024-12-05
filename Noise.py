import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from CNN import NeuralNet

def add_gaussian_noise(image, mean=0.0, std=0.5):
    noisy_img = image + torch.randn(image.size()) * std + mean
    noisy_img = torch.clamp(noisy_img, 0., 1.)
    return noisy_img

def add_salt_pepper_noise(image, prob=0.2):
    noisy_img = image.clone()
    num_pixels = image.numel()
    num_salt = int(prob * num_pixels / 2)
    num_pepper = int(prob * num_pixels / 2)

    # salt
    coords = np.random.choice(num_pixels, num_salt, replace=False)
    noisy_img.view(-1)[coords] = 1.0

    # pepper
    coords = np.random.choice(num_pixels, num_pepper, replace=False)
    noisy_img.view(-1)[coords] = 0.0

    return noisy_img

def Add_Noise(noise_type, mean=0.0, std=0.5, prob=0.2):
    output_lines = []

    #device = torch.device("mps")
    #device = torch.device("cuda")
    device = torch.device("cpu")

    model = NeuralNet().to(device)
    if not os.path.exists("mnist_cnn.pt"):
        message = "Trained model 'mnist_cnn.pt' not found. Run CNN.py first"
        print(message)
        output_lines.append(message)
        return output_lines

    model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
    model.eval()

    transform = transforms.Compose([transforms.ToTensor(),])

    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)

    normalize = transforms.Normalize((0.1307,), (0.3081,))
    

    correct_images = []
    correct_labels = []
    with torch.no_grad():
        for data, target in test_dataset:
            data, target = data.to(device), torch.tensor(target).to(device)

            normalized_data = normalize(data)

            output = model(normalized_data.unsqueeze(0))
            pred = output.argmax(dim=1, keepdim=True)
            if pred.item() == target.item():
                correct_images.append(data.cpu())
                correct_labels.append(target.item())
                if len(correct_images) == 2:
                    break

    if len(correct_images) < 2:
        message = "Not enough correctly classified images found."
        print(message)
        output_lines.append(message)
        return output_lines

    results = ""

    for idx in range(2):
        image = correct_images[idx]
        label = correct_labels[idx]

        if noise_type.lower() == "gaussian":
            noisy_image = add_gaussian_noise(image, mean=mean, std=std)
            noise_param = f"Gaussian Noise (mean={mean}, std={std})"
        elif noise_type.lower() in ["s&p", "salt_pepper", "salt and pepper"]:
            noisy_image = add_salt_pepper_noise(image, prob=prob)
            noise_param = f"Salt and Pepper Noise (prob={prob})"

        norm_noisy = normalize(noisy_image)

        output = model(norm_noisy.unsqueeze(0).to(device))
        pred = output.argmax(dim=1, keepdim=True).item()
        fooled = pred != label
        result = (f"Image {idx+1} - Original Label: {label}, "
                  f"Predicted After Noise: {pred} | Fooled: {fooled}")
        results += result + "\n"

        plt.imshow(noisy_image.squeeze(), cmap='gray')
        plt.title(f'Original: {label}, Predicted: {pred}')
        plt.savefig(f'noised_up_image_{idx+1}.png')
        plt.close()

    print(results)
    output_lines.append(results)
    return output_lines

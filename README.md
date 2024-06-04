# Retinal Scan Classification using EfficientNet-B0

## Overview

This project aims to leverage the EfficientNet-B0 architecture for the automated classification of retinal scans. The model has been fine-tuned to achieve high accuracy in identifying retinal conditions, which can assist in early diagnosis and treatment planning.
Dataset Link: https://www.kaggle.com/datasets/paultimothymooney/kermany2018
## Model Architecture

- **Base Model:** EfficientNet-B0
- **Customization:** Replaced the final fully connected layer to output four classes
- **Optimizer:** Adam
- **Loss Function:** Cross-Entropy Loss

**Prepare your dataset:**
    - Ensure your dataset is organized into subfolders for each class within the training and validation directories.

**Train the model:**

    ```python
    import torch
    import torch.nn as nn
    from efficientnet_pytorch import EfficientNet
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    train_dataset = datasets.ImageFolder('path_to_train_data', transform=transform)
    val_dataset = datasets.ImageFolder('path_to_val_data', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Define the model
    class EfficientNetB0(nn.Module):
        def __init__(self, num_classes=4):
            super(EfficientNetB0, self).__init__()
            self.model = EfficientNet.from_pretrained('efficientnet-b0')
            in_features = self.model._fc.in_features
            self.model._fc = nn.Linear(in_features, num_classes)

        def forward(self, x):
            return self.model(x)

    model = EfficientNetB0(num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    epochs = 20
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {(100 * correct / total):.2f}%')
        torch.save(model.state_dict(), 'Eff_Net.pth')
    ```

### Inference

1. **Load the model and perform inference on a single image:**

    ```python
    import torch
    from torchvision import transforms
    from PIL import Image

    # Load the trained model
    model = EfficientNetB0(num_classes=4)
    model.load_state_dict(torch.load('Eff_Net.pth'))
    model.eval()

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess the image
    image_path = 'path_to_image.jpeg'
    image = Image.open(image_path).convert("RGB")
    input_image = transform(image).unsqueeze(0)
    input_image = input_image.to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_image)
    _, predicted = torch.max(outputs, 1)
    class_index = predicted.item()

    # Map class index to class name
    class_names = {v: k for k, v in train_dataset.class_to_idx.items()}
    predicted_class_name = class_names[class_index]

    print(f'Predicted Class Index: {class_index}')
    print(f'Predicted Class Name: {predicted_class_name}')
    ```

## System Requirements

- **Hardware:** NVIDIA GPU (e.g., V100 or T4) recommended for training and inference.
- **Software:** Python 3.7+, PyTorch, EfficientNet-PyTorch, torchvision, PIL.

## Implementation Details

- **Training Time:** Approximately 4 hours on an NVIDIA GPU.
- **Compute Requirements:** Single GPU for training and inference.

## Data

- **Training Data:** Retinal scans labeled into four categories: DME, CNV, Drusen, and Normal.
- **Preprocessing:** Images resized to 224x224, normalized using ImageNet mean and standard deviation.

## Evaluation

- **Evaluation Metrics:** Accuracy, Sensitivity, Specificity.
- **Results:**
  - **Overall Accuracy:** 94.11%
  - **Overall Sensitivity:** 94.11%
  - **Overall Specificity:** 98.04%
  - Detailed class-wise metrics available in the repository.

## Results

- **Epoch-wise Performance:**
  - Training and validation loss and accuracy recorded for 20 epochs.
  - Final model achieves high accuracy and low validation loss.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

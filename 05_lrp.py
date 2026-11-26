# %% Imports
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import copy
import pandas as pd

# Set GPU device
print(torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Silicon / supported Mac
else:
    device = torch.device("cpu")
print("Using device:", device)

# %% Load data
TRAIN_ROOT = "data/brain_mri/training"
TEST_ROOT = "data/brain_mri/testing"
train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_ROOT)
test_dataset = torchvision.datasets.ImageFolder(root=TEST_ROOT)

# %% Check dataset information
print("Training path:", TRAIN_ROOT)
print("Testing path:", TEST_ROOT)

print("Number of training images:", len(train_dataset))
print("Number of testing images:", len(test_dataset))

print("Training class_to_idx:", train_dataset.class_to_idx)
print("Testing class_to_idx:", test_dataset.class_to_idx)

print("Class mappings match:",
      train_dataset.class_to_idx == test_dataset.class_to_idx)

assert train_dataset.class_to_idx == test_dataset.class_to_idx, \
    "Training and testing class mappings do not match!"

# %% Check duplicate images
import hashlib

def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

train_hashes = {
    get_file_hash(path)
    for path, label in train_dataset.samples
}

test_hashes = {
    get_file_hash(path)
    for path, label in test_dataset.samples
}

duplicates = train_hashes.intersection(test_hashes)

print("Unique training image hashes:", len(train_hashes))
print("Unique testing image hashes:", len(test_hashes))
print("Cross-set duplicate hashes:", len(duplicates))

if len(duplicates) == 0:
    print("No exact duplicate files found.")
else:
    print("Warning: Duplicate files detected!")

print("===== Duplicate check completed =====")
# %% Building the model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)

        # Replace output layer according to our problem
        in_feats = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(in_feats, 4)

    def forward(self, x):
        x = self.vgg16(x)
        return x

model = CNNModel()
model.to(device)
model

# %% Prepare data for pretrained model
train_dataset = torchvision.datasets.ImageFolder(
        root=TRAIN_ROOT,
        transform=transforms.Compose([
                      transforms.Resize((255,255)),
                      transforms.ToTensor()
        ])
)

test_dataset = torchvision.datasets.ImageFolder(
        root=TEST_ROOT,
        transform=transforms.Compose([
                      transforms.Resize((255,255)),
                      transforms.ToTensor()
        ])
)

# train_dataset[0][0].permute(1,2,0)

# %% Create data loaders
batch_size = 4
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True
)

# %% Train
cross_entropy_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
epochs = 10

# Iterate x epochs over the train data
model.train()
for epoch in range(epochs):
    for i, batch in enumerate(train_loader, 0):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # Labels are automatically one-hot-encoded
        loss = cross_entropy_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Batch [{i + 1}/{len(train_loader)}] "
              f"Loss: {loss.item():.4f}")

# %% Inspect predictions for first batch
import pandas as pd
inputs, labels = next(iter(test_loader))
inputs = inputs.to(device)
labels = labels.numpy()
model.eval()
with torch.no_grad():
    outputs = model(inputs).max(1).indices.cpu().numpy()
comparison = pd.DataFrame()
print("Batch accuracy: ", (labels == outputs).sum() / len(labels))
comparison["labels"] = labels

comparison["outputs"] = outputs
comparison

# %% Layerwise relevance propagation for VGG16
# For other CNN architectures this code might become more complex
# Source: https://git.tu-berlin.de/gmontavon/lrp-tutorial
# http://iphome.hhi.de/samek/pdf/MonXAI19.pdf

def new_layer(layer, g):
    """Clone a layer and pass its parameters through the function g."""
    layer = copy.deepcopy(layer)
    try: layer.weight = torch.nn.Parameter(g(layer.weight))
    except AttributeError: pass
    try: layer.bias = torch.nn.Parameter(g(layer.bias))
    except AttributeError: pass
    return layer

def dense_to_conv(layers):
    """ Converts a dense layer to a conv layer """
    newlayers = []
    for i,layer in enumerate(layers):
        if isinstance(layer, nn.Linear):
            newlayer = None
            if i == 0:
                m, n = 512, layer.weight.shape[0]
                newlayer = nn.Conv2d(m,n,7)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n,m,7,7))
            else:
                m,n = layer.weight.shape[1],layer.weight.shape[0]
                newlayer = nn.Conv2d(m,n,1)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n,m,1,1))
            newlayer.bias = nn.Parameter(layer.bias)
            newlayers += [newlayer.to(layer.weight.device)]
        else:
            newlayers += [layer]
    return newlayers


def get_linear_layer_indices(model):
    offset = len(model.vgg16._modules['features']) + 1
    indices = []
    for i, layer in enumerate(model.vgg16._modules['classifier']):
        if isinstance(layer, nn.Linear):
            indices.append(i)
    indices = [offset + val for val in indices]
    return indices


def apply_lrp_on_vgg16(model, image):
    image = torch.unsqueeze(image, 0)

    layers = list(model.vgg16._modules['features']) \
             + [model.vgg16._modules['avgpool']] \
             + dense_to_conv(list(model.vgg16._modules['classifier']))

    n_layers = len(layers)
    activations = [image] + [None] * n_layers  # list of activations

    with torch.no_grad():
        for layer in range(n_layers):
            # Keep the 4D shape for the converted convolutional classifier.
            activations[layer + 1] = layers[layer].forward(activations[layer])

    output_activation = activations[-1].detach()
    target = output_activation.flatten(1).argmax(dim=1).item()
    one_hot_output = torch.zeros_like(output_activation)
    one_hot_output[:, target] = output_activation[:, target]
    activations[-1] = one_hot_output

    relevances = [None] * n_layers + [activations[-1]]
    # Iterate over the layers in reverse order
    for layer in range(0, n_layers)[::-1]:
        current = layers[layer]
        # Treat max pooling layers as avg pooling
        if isinstance(current, torch.nn.MaxPool2d):
            layers[layer] = torch.nn.AvgPool2d(2)
            current = layers[layer]
        if isinstance(current, torch.nn.Conv2d) or \
                isinstance(current, torch.nn.AvgPool2d) or \
                isinstance(current, torch.nn.Linear) or \
                isinstance(current, torch.nn.AdaptiveAvgPool2d):
            activations[layer] = activations[layer].detach().requires_grad_(True)

            # Apply variants of LRP depending on the depth
            # see: https://link.springer.com/chapter/10.1007%2F978-3-030-28954-6_10
            # Lower layers, LRP-gamma >> Favor positive contributions (activations)
            if layer <= 16:       rho = lambda p: p + 0.25 * p.clamp(min=0)
            # Middle layers, LRP-epsilon >> Remove some noise / Only most salient factors survive
            if 17 <= layer <= 30: rho = lambda p: p
            # Upper Layers, LRP-0 >> Basic rule
            if layer >= 31:       rho = lambda p: p

            # Transform weights of layer and execute forward pass
            # Sign-aware stabilizer avoids division by zero for negative activations.
            z_raw = new_layer(layers[layer], rho).forward(activations[layer])
            epsilon = 1e-9
            if 17 <= layer <= 30:
                epsilon = epsilon + 0.25 * z_raw.detach().square().mean().sqrt()
            z = z_raw + torch.where(z_raw >= 0, torch.ones_like(z_raw), -torch.ones_like(z_raw)) * epsilon
            # Element-wise division between relevance of the next layer and z
            s = (relevances[layer + 1] / z).detach()
            # Calculate the gradient and multiply it by the activation
            c = torch.autograd.grad((z * s).sum(), activations[layer])[0]
            # Assign new relevance values
            relevances[layer] = (activations[layer] * c).detach()
        else:
            relevances[layer] = relevances[layer + 1]

    return relevances[0]


# %%
# Calculate relevances for first image in this test batch
# Use CPU for XAI on Mac to avoid unsupported MPS backward operations.
if device.type == "mps":
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()  # Release Adam state before CPU explanation
    model = model.to("cpu")
    inputs = inputs.to("cpu")
    device = torch.device("cpu")
    torch.mps.empty_cache()
correct_indices = np.where(outputs == labels)[0]

if len(correct_indices) > 0:
    image_id = int(correct_indices[0])
else:
    image_id = 0

print("Selected image ID:", image_id)
image_relevances = apply_lrp_on_vgg16(model, inputs[image_id])
image_relevances = image_relevances.permute(0, 2, 3, 1).detach().cpu().numpy()[0]
# Sum RGB relevance and preserve the sign around zero.
image_relevances = image_relevances.sum(axis=2)
image_relevances = image_relevances / max(np.abs(image_relevances).max(), 1e-12)
# Show relevances
pred_label = list(test_dataset.class_to_idx.keys())[
    list(test_dataset.class_to_idx.values())
    .index(labels[image_id])]
if outputs[image_id] == labels[image_id]:
    print("Groundtruth for this image: ", pred_label)

    # Plot images next to each other
    plt.axis('off')
    plt.subplot(1, 2, 1)
    plt.imshow(image_relevances, cmap="seismic", vmin=-1, vmax=1)
    plt.subplot(1, 2, 2)
    plt.imshow(inputs[image_id].permute(1, 2, 0).detach().cpu().numpy())
    plt.show()
else:
    print("This image is not classified correctly.")


# %%
print("Predicted class index:", int(outputs[image_id]))
print("Predicted class name:",
      test_dataset.classes[int(outputs[image_id])])

raw_relevance = apply_lrp_on_vgg16(model, inputs[image_id])

print("Raw relevance shape:", raw_relevance.shape)
print("Raw relevance minimum:", raw_relevance.min().item())
print("Raw relevance maximum:", raw_relevance.max().item())
print("Raw relevance sum:", raw_relevance.sum().item())
print("All values finite:",
      torch.isfinite(raw_relevance).all().item())
# %% Grad-CAM for VGG16 (second XAI method)
def apply_gradcam_on_vgg16(model, image):
    model.eval()
    image = image.unsqueeze(0).detach().requires_grad_(True)
    # ReLU immediately after the last VGG16 convolution, before max pooling.
    target_layer = model.vgg16.features[29]
    feature_maps = []

    def forward_hook(module, inputs, output):
        feature_maps.append(output)

    handle = target_layer.register_forward_hook(forward_hook)
    try:
        with torch.enable_grad():
            scores = model(image)
            target = scores.argmax(dim=1).item()
            features = feature_maps[0]
            gradients = torch.autograd.grad(scores[0, target], features)[0]
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * features).sum(dim=1, keepdim=True))
            cam = torch.nn.functional.interpolate(
                cam, size=image.shape[-2:], mode="bilinear", align_corners=False
            )[0, 0]
            cam = cam / cam.max().clamp_min(1e-12)
            return cam.detach().cpu().numpy()
    finally:
        handle.remove()


# %% Compare LRP and Grad-CAM on the same test image
image_gradcam = apply_gradcam_on_vgg16(model, inputs[image_id])
original_image = inputs[image_id].permute(1, 2, 0).detach().cpu().numpy()
class_names = test_dataset.classes
true_class = class_names[int(labels[image_id])]
predicted_class = class_names[int(outputs[image_id])]

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title("Original MRI")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(image_relevances, cmap="seismic", vmin=-1, vmax=1)
plt.title("LRP")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(original_image)
plt.imshow(image_gradcam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
plt.title("Grad-CAM")
plt.axis("off")

plt.suptitle(f"Ground truth: {true_class} | Prediction: {predicted_class}")
plt.tight_layout()
plt.savefig("lrp_gradcam_comparison.png", dpi=200, bbox_inches="tight")
plt.show()

# %% Grad-CAM execution evidence
print("===== Grad-CAM execution evidence =====")

print("Selected image ID:", image_id)
print("Ground truth:", test_dataset.classes[int(labels[image_id])])
print("Predicted class index:", int(outputs[image_id]))
print("Predicted class name:",
      test_dataset.classes[int(outputs[image_id])])

print("Target layer:", model.vgg16.features[29])
print("Grad-CAM output shape:", image_gradcam.shape)
print("Grad-CAM minimum:", image_gradcam.min())
print("Grad-CAM maximum:", image_gradcam.max())
print("Grad-CAM mean:", image_gradcam.mean())
print("All values finite:", np.isfinite(image_gradcam).all())

# %% Evaluate the complete test dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

model.eval()
eval_device = next(model.parameters()).device

all_labels = []
all_predictions = []

with torch.no_grad():
    for test_inputs, test_labels in test_loader:
        test_inputs = test_inputs.to(eval_device)

        test_outputs = model(test_inputs)
        predictions = test_outputs.argmax(dim=1)

        all_labels.extend(test_labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

accuracy = accuracy_score(all_labels, all_predictions)

precision = precision_score(
    all_labels, all_predictions,
    average="macro", zero_division=0
)

recall = recall_score(
    all_labels, all_predictions,
    average="macro", zero_division=0
)

f1 = f1_score(
    all_labels, all_predictions,
    average="macro", zero_division=0
)

print("===== Full test evaluation =====")
print("Test images evaluated:", len(all_labels))
print("Accuracy:", accuracy)
print("Macro Precision:", precision)
print("Macro Recall:", recall)
print("Macro F1-score:", f1)

print("\nClassification report:")
print(classification_report(
    all_labels,
    all_predictions,
    labels=list(range(len(test_dataset.classes))),
    target_names=test_dataset.classes,
    zero_division=0
))

# %% Confusion matrix
cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(test_dataset.classes)))
)

print("Confusion matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(8, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=test_dataset.classes
)

disp.plot(
    cmap="Blues",
    values_format="d",
    xticks_rotation=45,
    ax=ax
)

plt.title("VGG16 Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# %% Case A prediction score
model.eval()

with torch.no_grad():
    case_scores = model(
        inputs[image_id].unsqueeze(0).to(
            next(model.parameters()).device
        )
    )

    case_target = case_scores.argmax(dim=1).item()
    case_logit = case_scores[0, case_target].item()

print("Case ID: A")
print("Batch image index:", image_id)

print("Dataset label:",
      test_dataset.classes[int(labels[image_id])])

print("Predicted class:",
      test_dataset.classes[case_target])

print("Predicted class index:", case_target)
print("Predicted class logit:", case_logit)
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json

with open("./classes.json", 'r') as f:
    CLASSES = json.load(f)
# load model
class block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels,intermediate_channels,kernel_size=1,stride=1,padding=0,bias=False,)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(intermediate_channels,intermediate_channels,kernel_size=3,stride=stride,padding=1,bias=False,)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(intermediate_channels,intermediate_channels * self.expansion,kernel_size=1,stride=1,padding=0,bias=False,)
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class ImageClassificationBase(nn.Module):
    def training_step(self, images,labels):
        #images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return loss,acc
        #return loss
class ResNet(ImageClassificationBase):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels,intermediate_channels * 4,kernel_size=1,stride=stride, bias=False,),
                nn.BatchNorm2d(intermediate_channels * 4),)

        layers.append(block(self.in_channels, intermediate_channels, identity_downsample, stride))

        self.in_channels = intermediate_channels * 4

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)

model = ResNet50(num_classes=38) 
model.load_state_dict(torch.load("./RESNET50-MODEL/pretrained_weights.pth", map_location=torch.device('cpu')))
model.eval()

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),

    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Inference function
def predict_image(image_path, temperature=2.0):
    input_image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_image)
        # Apply temperature scaling to logits
        scaled_logits = output[0] / temperature
        # Add debugging information
        probabilities = torch.nn.functional.softmax(scaled_logits, dim=0)
        # Get top 3 predictions
        top3_prob, top3_indices = torch.topk(probabilities, 3)
        print("\nTop 3 predictions:")
        for i in range(3):
            print(f"{CLASSES[top3_indices[i]]}: {top3_prob[i].item()*100:.2f}%")
        print("\nRaw output shape:", output.shape)
        print("Raw output values:", output[0][:5])  # Print first 5 logits
        print("Temperature-scaled logits:", scaled_logits[:5])  # Print scaled logits
    _, preds = torch.max(scaled_logits.unsqueeze(0), 1)
    return CLASSES[preds[0].item()]

# Example usage
if __name__ == "__main__":
    image_path = "./RESNET50-MODEL/apple_scab.JPG"
    print(f"\nProcessing image: {image_path}")
    prediction = predict_image(image_path)
    print(f"\nFinal prediction: {prediction}")
    print(f"Predicted disease/condition: {prediction}")
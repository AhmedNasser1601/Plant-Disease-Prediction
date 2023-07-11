# from flask import render_template, request, jsonify
# from app import app

# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# import torch.nn.functional as F
# from PIL import Image
# import io

# def accuracy(outputs, labels):
#     _, preds = torch.max(outputs, dim=1)
#     return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# class ImageClassificationBase(nn.Module):
#     def training_step(self, batch):
#         images, labels = batch 
#         out = self(images)                  # Generate predictions
#         loss = F.cross_entropy(out, labels) # Calculate loss
#         return loss
    
#     def validation_step(self, batch):
#         images, labels = batch 
#         out = self(images)                    # Generate predictions
#         loss = F.cross_entropy(out, labels)   # Calculate loss
#         acc = accuracy(out, labels)           # Calculate accuracy
#         return {'val_loss': loss.detach(), 'val_acc': acc}
        
#     def validation_epoch_end(self, outputs):
#         batch_losses = [x['val_loss'] for x in outputs]
#         epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
#         batch_accs = [x['val_acc'] for x in outputs]
#         epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
#         return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
#     def epoch_end(self, epoch, result):
#         print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
#             epoch, result['train_loss'], result['val_loss'], result['val_acc']))

# class Plant_Disease_Model(ImageClassificationBase):
#     def __init__(self):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2,2), 

#             nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2,2), 

#             nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2,2), 

#             nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2,2),

#             nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2,2), 
#             nn.AdaptiveMaxPool2d(1),

#             nn.Flatten(),
#             nn.Linear(1024,512),
#             nn.ReLU(),
#             nn.Linear(512,256),
#             nn.ReLU(),
#             nn.Linear(256,14)
#         )
#     def forward(self, x):
#         return self.network(x)
    

# transform=transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Resize((250, 250)),
#                 transforms.Normalize(mean=[0.4757,0.5001,0.4264],std=[0.2166,0.1957,0.2322]),                  
# ])

# classes_name =['Apple___Apple_scab',
#  'Apple___Black_rot',
#  'Apple___Cedar_apple_rust',
#  'Apple___healthy',
#  'Grape___Black_rot',
#  'Grape___Esca_(Black_Measles)',
#  'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
#  'Grape___healthy',
#  'Orange___Haunglongbing_(Citrus_greening)',
#  'Peach___Bacterial_spot',
#  'Peach___healthy',
#  'Raspberry___healthy',
#  'Strawberry___Leaf_scorch',
#  'Strawberry___healthy']

# model = Plant_Disease_Model()
# model.load_state_dict(torch.load('fruitDisease-vgg16-modified-25epoch.pth', map_location=torch.device('cpu')))
# model.eval()

# def predict_img(img):
#     image = Image.open(io.BytesIO(img))
#     image = transform(image)
#     xb = image.unsqueeze(0)
#     yb = model(xb)
#     _, preds = torch.max(yb, dim=1)
#     return classes_name[preds[0].item()]


# @app.route('/login')
# def Home():
#     return "blablablalblablab" 


# @app.route('/predict', methods=['POST'])
# def predict():
#     image = request.files['image'].read()
#     prediction = predict_img(image)
#     print(prediction)
#     return "blablabla"
    

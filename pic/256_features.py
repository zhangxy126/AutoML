import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

class InceptionV3(nn.Module):
    def __init__(self, output_dim=256):
        super(InceptionV3, self).__init__()
        self.inception = models.inception_v3(pretrained=True, aux_logits=True)  
        for param in self.inception.parameters():
            param.requires_grad = False 
        self.inception.fc = nn.Sequential(
            nn.Linear(self.inception.fc.in_features, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.inception.eval()  

    def extract_features(self, image_tensor):
        with torch.no_grad():
            output = self.inception(image_tensor)
            return output if isinstance(output, torch.Tensor) else output[0]  
        
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features_from_datasets(root_folder):
    feature_extractor = InceptionV3()  
    features_dict = {}  

    dataset_folders = [f for f in sorted(os.listdir(root_folder)) if os.path.isdir(os.path.join(root_folder, f))]

    for dataset_name in tqdm(dataset_folders, desc="Processing datasets"):
        dataset_path = os.path.join(root_folder, dataset_name)
        image_paths = sorted([os.path.join(dataset_path, img) for img in os.listdir(dataset_path) if img.endswith('.png')])

        dataset_features = []  

        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert("RGB")
                image = transform(image) 
                image_features = feature_extractor.extract_features(image.unsqueeze(0)).squeeze(0)  
                dataset_features.append(image_features)
            except Exception as e:
                print(f"Warning: Failed to process image {image_path}, error: {e}")

        if not dataset_features:
            print(f"Warning: No valid images found for dataset {dataset_name}, skipping...")
            continue

        features_dict[dataset_name] = torch.stack(dataset_features)  

    return features_dict

root_folder = "t-SNE_pictures(20)"
features_dict = extract_features_from_datasets(root_folder)

torch.save(features_dict, "256image_features.pt")
print("Features have been extracted and saved as '256image_features.pt'.")

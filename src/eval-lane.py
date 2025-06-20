import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score
from tqdm import tqdm
import os
from PIL import Image
from lane_detector import valLaneDetector
from mymodel.enetpytorch import ENet

# Set the environment variable for CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = CBR(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = CBR(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = CBR(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = CBR(128, 64)

        self.conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return torch.sigmoid(self.conv(dec1))


# Define transformations for the images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))  # Resize to match the input size of the model
])


# Custom dataset class
class BDD100KDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_list = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.image_list[idx].replace('.jpg', '.png'))

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# Load the trained model
model = UNet(3, 1).cuda()
model.load_state_dict(
    torch.load('./best_unet_lane_detection.pth', map_location=torch.device('cuda')))  # ['model_state_dict'])
model.eval()

# Define loss function
criterion = nn.BCEWithLogitsLoss()

# Load validation data
val_image_dir = rf'D:\BaiduNetdiskDownload\images\val'
val_mask_dir = rf'D:\BaiduNetdiskDownload\masks\val'
val_dataset = BDD100KDataset(val_image_dir, val_mask_dir, transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# Evaluation function
def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    val_f1 = 0.0
    val_jaccard = 0.0
    val_precision = 0.0
    val_recall = 0.0
    val_balanced_accuracy = 0.0
    val_dice = 0.0
    val_avg_precision = 0.0
    val_avg_recall = 0.0
    num_batches = len(val_loader)

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc='Evaluating'):
            images = images.cuda()
            masks = masks.cuda()

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            preds = outputs.squeeze().cpu().numpy() > 0.5
            true = masks.cpu().numpy() > 0.5

            val_accuracy += accuracy_score(true.flatten(), preds.flatten())
            val_f1 += f1_score(true.flatten(), preds.flatten())
            val_jaccard += jaccard_score(true.flatten(), preds.flatten())
            val_precision += precision_score(true.flatten(), preds.flatten())
            val_recall += recall_score(true.flatten(), preds.flatten())
            val_balanced_accuracy += balanced_accuracy_score(true.flatten(), preds.flatten())

            # Calculate Dice coefficient
            intersection = (preds * true).sum()
            union = preds.sum() + true.sum()
            dice = (2.0 * intersection) / (union + 1e-7)  # Add a small epsilon to avoid division by zero
            val_dice += dice

            # Calculate Average Precision and Average Recall
            # These metrics require multiple thresholds, so we approximate here
            # For a more accurate calculation, you would need to compute over multiple thresholds
            # Here we use a single threshold of 0.5
            val_avg_precision += precision_score(true.flatten(), preds.flatten())
            val_avg_recall += recall_score(true.flatten(), preds.flatten())

    val_loss /= num_batches
    val_accuracy /= num_batches
    val_f1 /= num_batches
    val_jaccard /= num_batches
    val_precision /= num_batches
    val_recall /= num_batches
    val_balanced_accuracy /= num_batches
    val_dice /= num_batches
    val_avg_precision /= num_batches
    val_avg_recall /= num_batches

    print(f'Validation Loss: {val_loss:.4f}')
    print(f'Validation Accuracy: {val_accuracy:.4f}')
    print(f'Validation F1 Score: {val_f1:.4f}')
    print(f'Validation Jaccard Score: {val_jaccard:.4f}')
    print(f'Validation Precision: {val_precision:.4f}')
    print(f'Validation Recall: {val_recall:.4f}')
    print(f'Validation Balanced Accuracy: {val_balanced_accuracy:.4f}')
    print(f'Validation Dice Coefficient: {val_dice:.4f}')
    print(f'Validation Average Precision: {val_avg_precision:.4f}')
    print(f'Validation Average Recall: {val_avg_recall:.4f}')


def evaluateml(val_loader):
    val_accuracy = 0.0
    val_f1 = 0.0
    val_jaccard = 0.0
    val_precision = 0.0
    val_recall = 0.0
    val_balanced_accuracy = 0.0
    val_dice = 0.0
    val_avg_precision = 0.0
    val_avg_recall = 0.0
    num_batches = len(val_loader)
    DL = valLaneDetector()
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc='Evaluating'):
            outputs = []
            for img in images:
                # Convert tensor to numpy and transpose to (H, W, C) format
                img_np = img.cpu().numpy().transpose(1, 2, 0)
                # Ensure the pixel values are in the range [0, 255]
                if img_np.dtype != np.uint8:
                    img_np = (img_np * 255).astype(np.uint8)
                # Process the image to get lane mask
                lane_mask = DL.process(img_np)
                # Convert the lane mask to tensor and add to outputs
                outputs.append(lane_mask)
            # Convert outputs to numpy array then to tensor
            outputs = np.array(outputs)
            outputs = torch.from_numpy(outputs).unsqueeze(1)
            masks = masks
            preds = outputs.cpu().numpy() > 0.5
            true = masks.cpu().numpy() > 0.5

            val_accuracy += accuracy_score(true.flatten(), preds.flatten())
            val_f1 += f1_score(true.flatten(), preds.flatten())
            val_jaccard += jaccard_score(true.flatten(), preds.flatten())
            val_precision += precision_score(true.flatten(), preds.flatten())
            val_recall += recall_score(true.flatten(), preds.flatten())
            val_balanced_accuracy += balanced_accuracy_score(true.flatten(), preds.flatten())

            # Calculate Dice coefficient
            intersection = (preds * true).sum()
            union = preds.sum() + true.sum()
            dice = (2.0 * intersection) / (union + 1e-7)  # Add a small epsilon to avoid division by zero
            val_dice += dice

            val_avg_precision += precision_score(true.flatten(), preds.flatten())
            val_avg_recall += recall_score(true.flatten(), preds.flatten())

    val_accuracy /= num_batches
    val_f1 /= num_batches
    val_jaccard /= num_batches
    val_precision /= num_batches
    val_recall /= num_batches
    val_balanced_accuracy /= num_batches
    val_dice /= num_batches
    val_avg_precision /= num_batches
    val_avg_recall /= num_batches

    print(f'Validation Accuracy: {val_accuracy:.4f}')
    print(f'Validation F1 Score: {val_f1:.4f}')
    print(f'Validation Jaccard Score: {val_jaccard:.4f}')
    print(f'Validation Precision: {val_precision:.4f}')
    print(f'Validation Recall: {val_recall:.4f}')
    print(f'Validation Balanced Accuracy: {val_balanced_accuracy:.4f}')
    print(f'Validation Dice Coefficient: {val_dice:.4f}')
    print(f'Validation Average Precision: {val_avg_precision:.4f}')
    print(f'Validation Average Recall: {val_avg_recall:.4f}')


# evaluate(model, val_loader, criterion)
evaluateml(val_loader)
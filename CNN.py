import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models


def compute_resize_and_pad(h, w, target_size=(256, 256)):
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    return scale, pad_x, pad_y, new_w, new_h

def resize_and_pad_frame(frame, scale, pad_x, pad_y, new_w, new_h, target_size=(256, 256)):
    resized = cv2.resize(frame, (new_w, new_h))
    padded = np.zeros((target_size[1], target_size[0], 3), dtype=resized.dtype)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return padded


def get_video_label_pairs(video_base, label_base):

    pairs = []
    for root, _, files in os.walk(video_base):
        for file in files:
            if file.lower().endswith(".webm"):
                video_path = os.path.join(root, file)
                rel_path = os.path.relpath(video_path, video_base)
                rel_dir = os.path.splitext(rel_path)[0]
                pelvis_label_path = os.path.join(label_base, rel_dir, "COCO", "2D_pelvis.txt")
                if os.path.exists(pelvis_label_path):
                    pairs.append((video_path, pelvis_label_path))
                    print(f"Found pair: {video_path} <--> {pelvis_label_path}")

    return pairs


def load_keypoints(label_file, expected_values_per_line=None):

    keypoints_list = []
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            first_token = line.split(";")[0].replace(",", ".")
            if not first_token.replace(".", "", 1).replace("-", "", 1).isdigit():
                continue
            line = line.replace(",", ".")
            parts = [token for token in line.split(";") if token]
            if expected_values_per_line and len(parts) != expected_values_per_line:
                print(f"Warning: Expected {expected_values_per_line} values but found {len(parts)}. Skipping line.")
                continue
            try:
                values = [float(x) for x in parts]
            except ValueError as e:
                print(f"Could not convert line to floats: {line}\nError: {e}")
                continue
            try:
                keypoints_array = np.array(values).reshape(-1, 3)
            except ValueError as e:
                print(f"Error reshaping values from line: {line}\nError: {e}")
                continue
            keypoints_list.append(keypoints_array)
    print(f"Loaded {len(keypoints_list)} keypoint entries from {label_file}")
    return keypoints_list


class AllSwimPelvisDataset(Dataset):
    def __init__(self, video_label_pairs, transform=None, target_size=(256, 256)):

        self.transform = transform
        self.samples = []

        for video_path, pelvis_label_path in video_label_pairs:
            print(f"[Cache] Processing video: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Could not open {video_path}. Skipping.")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ret, frame0 = cap.read()
            if not ret or frame0 is None:
                print(f"No frames in {video_path}. Skipping.")
                cap.release()
                continue

            # Compute scale/pad from the first frame
            h, w = frame0.shape[:2]
            scale, pad_x, pad_y, new_w, new_h = compute_resize_and_pad(h, w, target_size=target_size)

            # Reset capture to the first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Load pelvis keypoints only
            pelvis_kp_list = load_keypoints(pelvis_label_path, expected_values_per_line=54)
            min_len = min(total_frames, len(pelvis_kp_list))
            if min_len == 0:
                print(f"Skipping {video_path}; no matching frames/keypoints found.")
                cap.release()
                continue
            print(f"Using {min_len} frames from {video_path}")


            for i in range(min_len):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break


                frame_padded = resize_and_pad_frame(frame, scale, pad_x, pad_y, new_w, new_h, target_size=target_size)
                frame_rgb = cv2.cvtColor(frame_padded, cv2.COLOR_BGR2RGB)


                pelvis_keypoints = pelvis_kp_list[i][:17, :2]

                transformed_kp = (pelvis_keypoints * scale + np.array([pad_x, pad_y])) / target_size[0]

                self.samples.append((frame_rgb, transformed_kp))

            cap.release()
        print(f"Total frames cached in memory: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_rgb, keypoints = self.samples[idx]
        if self.transform:
            img_t = self.transform(frame_rgb)
        else:
            img_t = transforms.ToTensor()(frame_rgb)
        keypoints_t = torch.tensor(keypoints, dtype=torch.float32)
        return img_t, keypoints_t

#Res net 18 cnn
class PoseResNet(nn.Module):
    def __init__(self, num_keypoints=17):
        super(PoseResNet, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_keypoints * 2)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        x = x.view(x.size(0), -1, 2)

        mean = x.mean(dim=1, keepdim=True)
        x = x + (0.5 - mean)
        return x

#training function
def train_model(dataset, num_epochs=15, batch_size=256):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    model = PoseResNet(num_keypoints=17)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, gt_keypoints in dataloader:
            images = images.to(device, non_blocking=True)
            gt_keypoints = gt_keypoints.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred_keypoints = model(images)
            loss = criterion(pred_keypoints, gt_keypoints)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}")

    save_path = "pose_cnn_model6.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")



def main():
    video_base = r"D:\SwimmingDataset\Aerial"
    label_base = r"D:\SwimmingDataset\Freestyle\Aerial"


    video_label_pairs = get_video_label_pairs(video_base, label_base)
    if not video_label_pairs:
        print("No video/label pairs found.")
        return


    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.ToTensor()
    ])


    dataset = AllSwimPelvisDataset(
        video_label_pairs,
        transform=train_transform,
        target_size=(256, 256)
    )

    if len(dataset) == 0:
        print("Empty dataset.")
        return


    train_model(dataset, num_epochs=15, batch_size=256)

if __name__ == "__main__":
    main()

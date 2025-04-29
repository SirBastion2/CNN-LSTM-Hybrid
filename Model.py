import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt




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

def load_pretrained_cnn(model_path="pose_cnn_model.pth",
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    cnn = PoseResNet(num_keypoints=17)
    cnn.load_state_dict(torch.load(model_path, map_location=device))
    cnn.to(device)
    cnn.eval()
    for param in cnn.parameters():
        param.requires_grad = False
    return cnn




def compute_pseudo_fatigue_labels(kp_seq, baseline_frames=10):
    T = kp_seq.shape[0]
    stroke_length = 1.93  # in meters

    stroke_rates = np.zeros(T)
    split_times = np.zeros(T)
    v_split = np.zeros(T)
    E = np.zeros(T)
    A = np.zeros(T)

    for t in range(T):
        t_frac = t / (T - 1) if T > 1 else 0
        stroke_rates[t] = 50 - 10 * t_frac
        split_times[t] = 15 + 2 * t_frac
        v_split[t] = 25 / split_times[t]
        E[t] = (v_split[t] * stroke_length * 60) / stroke_rates[t]
        A[t] = np.abs(kp_seq[t, 11, 0] - kp_seq[t, 12, 0])

    baseline_E = np.mean(E[:baseline_frames])
    baseline_stroke_rate = np.mean(stroke_rates[:baseline_frames])
    baseline_v = np.mean(v_split[:baseline_frames])
    baseline_A = np.mean(A[:baseline_frames])

    deltaE = (baseline_E - E) / baseline_E
    deltaK = (baseline_stroke_rate - stroke_rates) / baseline_stroke_rate
    deltaS = (baseline_v - v_split) / baseline_v
    deltaA = (A - baseline_A) / baseline_A

    P = 0.60 * (1 - deltaE) + 0.15 * (1 - deltaK) + 0.15 * (1 / (1 + deltaA)) + 0.10 * (1 - deltaS)
    fatigue = 100 * (1 - P)
    fatigue = np.maximum(fatigue, 0)
    return fatigue




class SwimSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, video_path, cnn_model, transform=None, target_size=(256, 256), fps=30):
        self.video_path = video_path
        self.cnn_model = cnn_model
        self.transform = transform
        self.target_size = target_size
        self.fps = fps
        self.features, self.pseudo_labels = self._process_video()

    def _process_video(self):
        frames = []
        if os.path.isdir(self.video_path):
            video_files = sorted([os.path.join(self.video_path, f) for f in os.listdir(self.video_path)
                                  if f.lower().endswith(('.mp4', '.avi', '.webm', '.mov'))])
            if not video_files:
                raise ValueError(f"No video files found in directory: {self.video_path}")
            for video_file in video_files:
                cap = cv2.VideoCapture(video_file)
                if not cap.isOpened():
                    print(f"Warning: Could not open video file: {video_file}")
                    continue
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    h, w = frame.shape[:2]
                    scale, pad_x, pad_y, new_w, new_h = compute_resize_and_pad(h, w, target_size=self.target_size)
                    frame_padded = resize_and_pad_frame(frame, scale, pad_x, pad_y, new_w, new_h, target_size=self.target_size)
                    frame_rgb = cv2.cvtColor(frame_padded, cv2.COLOR_BGR2RGB)
                    if self.transform:
                        frame_rgb = self.transform(frame_rgb)
                    else:
                        frame_rgb = transforms.ToTensor()(frame_rgb)
                    frames.append(frame_rgb)
                cap.release()
        else:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {self.video_path}")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]
                scale, pad_x, pad_y, new_w, new_h = compute_resize_and_pad(h, w, target_size=self.target_size)
                frame_padded = resize_and_pad_frame(frame, scale, pad_x, pad_y, new_w, new_h, target_size=self.target_size)
                frame_rgb = cv2.cvtColor(frame_padded, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame_rgb = self.transform(frame_rgb)
                else:
                    frame_rgb = transforms.ToTensor()(frame_rgb)
                frames.append(frame_rgb)
            cap.release()

        if len(frames) == 0:
            raise ValueError("No frames extracted from video(s).")

        video_tensor = torch.stack(frames)

        with torch.no_grad():
            features = self.cnn_model(video_tensor.to(next(self.cnn_model.parameters()).device))
            features = features.cpu().numpy()

        T = features.shape[0]
        flat_features = features.reshape(T, -1)
        pseudo_labels = compute_pseudo_fatigue_labels(features, baseline_frames=10)
        return flat_features, pseudo_labels.astype(np.float32)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        features = torch.tensor(self.features, dtype=torch.float32)
        labels = torch.tensor(self.pseudo_labels, dtype=torch.float32).unsqueeze(1)
        return features, labels




class FatigueLSTM(nn.Module):
    def __init__(self, input_dim=34, hidden_dim=64, num_layers=2, alpha=0.9):

        super(FatigueLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.alpha = alpha

    def forward(self, x, pseudo_labels):
        lstm_out, _ = self.lstm(x)
        correction = self.fc(lstm_out)
        correction = torch.relu(correction)
        fatigue_pred = self.alpha * pseudo_labels + (1 - self.alpha) * correction
        return fatigue_pred




def test_lstm():
    video_path = r"D:\SwimmingDataset\Videos\Untitled.mp4"
    cnn_model_path = "pose_cnn_model.pth"
    lstm_model_path = "fatigue_lstm_model.pth"
    fps = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn_model = load_pretrained_cnn(model_path=cnn_model_path, device=device)

    lstm_model = FatigueLSTM(input_dim=34, hidden_dim=64, num_layers=2, alpha=0.9)
    lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=device))
    lstm_model.to(device)
    lstm_model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    test_dataset = SwimSequenceDataset(video_path, cnn_model, transform=transform, target_size=(256, 256), fps=fps)
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for features, pseudo_labels in dataloader:
            features = features.to(device)
            pseudo_labels = pseudo_labels.to(device)
            outputs = lstm_model(features, pseudo_labels)
            fatigue_pred = outputs.squeeze(0).squeeze(-1).cpu().numpy()
            ground_truth = pseudo_labels.squeeze(0).squeeze(-1).cpu().numpy()
            break

    print("Predicted Fatigue Values:")
    print(fatigue_pred)
    print("\nPseudo Fatigue Labels (Ground Truth):")
    print(ground_truth)

    print("\nStatistics:")
    print("Predicted Fatigue - Mean: {:.2f}, Std: {:.2f}, Min: {:.2f}, Max: {:.2f}".format(
        np.mean(fatigue_pred), np.std(fatigue_pred), np.min(fatigue_pred), np.max(fatigue_pred)
    ))
    print("Pseudo Labels    - Mean: {:.2f}, Std: {:.2f}, Min: {:.2f}, Max: {:.2f}".format(
        np.mean(ground_truth), np.std(ground_truth), np.min(ground_truth), np.max(ground_truth)
    ))

    plt.figure(figsize=(10, 4))
    plt.plot(fatigue_pred, label='Predicted Fatigue')
    plt.plot(ground_truth, label='Pseudo Labels', linestyle='--')
    plt.xlabel("Frame")
    plt.ylabel("Fatigue Score")
    plt.title("Fatigue Prediction over Time")
    plt.ylim(0, 100)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_lstm()

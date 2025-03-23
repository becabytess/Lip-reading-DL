
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import gc
from torch.utils.data import Dataset, DataLoader
from typing import Any

# Constants
data_pth = '../datasets'
videos_pth = os.path.join(data_pth, 'videos')
aligns_pth = os.path.join(data_pth, 'alignments')

# vocab setup
vocab = 'abcdefghijklmnopqrstuvwxyz- '
vocab_size = len(vocab)
vti = {vocab[i]: i+1 for i in range(vocab_size)}
vti['-'] = 0  # using 0 for ctc loss blank
itv = {i: j for j, i in vti.items()}

class LipNetDataset(Dataset):
    def __init__(self, videos_pth, aligns_pth):
        self.aligns_pth = aligns_pth
        self.videos_pth = videos_pth
        self.speakers = os.listdir(videos_pth)
        videos = []
        aligns = []
        for speaker in self.speakers:
            video_files = os.listdir(os.path.join(self.videos_pth, speaker))
            align_files = os.listdir(os.path.join(self.aligns_pth, speaker))
            for video_file in video_files:
                is_valid_file = video_file.endswith('.mpg')
                align_file = video_file.replace('.mpg', '.align')
                if is_valid_file and align_file in align_files:
                    with open(os.path.join(self.aligns_pth, speaker, align_file), 'r') as f:
                        text = f.read()
                    if len(text) < 1:
                        continue
                    videos.append(os.path.join(self.videos_pth, speaker, video_file))
                    aligns.append(os.path.join(self.aligns_pth, speaker, align_file))

        self.videos = videos
        self.aligns = aligns

    def extract_text(self, align):
        with open(align, 'r') as f:
            text = f.read()
        text = ''.join(char for char in text if char in vocab)
        return [vti[char] for char in text]

    def extract_frames(self, video):
        frames = []
        cap = cv2.VideoCapture(video)

        while True:
            res, frame = cap.read()
            if not res:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
            frames.append(torch.tensor(frame, dtype=torch.float32))

        cap.release()
        return torch.stack(frames).float()

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        frames = self.extract_frames(video=self.videos[index])
        align = self.extract_text(self.aligns[index])
        align = torch.tensor(data=align, dtype=torch.long)
        return frames, align

def collate_fn(batch):
    videos = []
    labels = []
    for video, label in batch:
        if len(video) == 75:  # because the dataset contains some videos with less than 75 frames
            videos.append(video)
            labels.append(label)

    if not videos:
        return None, None

    return torch.stack(videos), list(labels)

class Model(nn.Module):
    def __init__(self, in_channels, frames, height, width, hidden_size=100):
        super().__init__()
        self.width = width
        self.height = height
        self.frames = frames
        self.in_channels = in_channels
        self.hidden_size = hidden_size


        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=(3,3,3),
                     stride=(1,1,1), padding=(1,1,1)),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0)),

            nn.Conv3d(in_channels=32, out_channels=16, kernel_size=(3,3,3),
                     stride=(1,1,1), padding=(1,1,1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0)),

            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3,3,3),
                     stride=(1,1,1), padding=(1,1,1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0))
        )


        self.cnn_output_size = (width//8) * (height//8) * 32


        self.forget = nn.Sequential(
            nn.Linear(in_features=self.cnn_output_size + hidden_size, out_features=hidden_size),
            nn.Sigmoid()
        )


        self.candidate = nn.Sequential(
            nn.Linear(in_features=hidden_size + self.cnn_output_size, out_features=hidden_size),
            nn.Tanh()
        )

        self.input = nn.Sequential(
            nn.Linear(in_features=hidden_size + self.cnn_output_size, out_features=hidden_size),
            nn.Sigmoid()
        )


        self.output = nn.Sequential(
            nn.Linear(hidden_size + self.cnn_output_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Sigmoid()
        )

        self.classifier = nn.Linear(2 * hidden_size, vocab_size)



    def bidirectional(self, X, isbackward=False):

        conv_out = self.conv(X)
        b, c, f, h, w = conv_out.size()
        conv_out = conv_out.contiguous().view(b, f, -1)


        cell_state = torch.zeros(b, self.hidden_size, device=X.device)
        hidden_state = torch.zeros(b, f, self.hidden_size, device=X.device)

        frame_range = range(f-1, -1, -1) if isbackward else range(f)

        for t in frame_range:
            xt = conv_out[:, t, :]
            prev_idx = t+1 if isbackward else t-1
            valid_prev = (prev_idx >= 0 and prev_idx < f)
            prev_hs = hidden_state[:, prev_idx, :] if valid_prev else torch.zeros(b, self.hidden_size, device=X.device)


            combined = torch.cat([xt, prev_hs], dim=1)


            forget_gate = self.forget(combined)
            input_gate = self.input(combined)
            candidate_gate = self.candidate(combined)
            output_gate = self.output(combined)


            new_cell_state = forget_gate * cell_state + input_gate * candidate_gate
            new_cell_state = torch.clamp(new_cell_state, -10, 10)
            new_hidden_state = output_gate * torch.tanh(new_cell_state)

            cell_state = new_cell_state
            hidden_state[:, t, :] = new_hidden_state

        return hidden_state

    def forward(self, X):

        forward_hidden = self.bidirectional(X)
        backward_hidden = self.bidirectional(X, isbackward=True)
        hidden_state = torch.cat([forward_hidden, backward_hidden], dim=2)
        logits = self.classifier(hidden_state)
        return nn.functional.log_softmax(logits, dim=2)

def clear_gpu_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


dataset = LipNetDataset(videos_pth, aligns_pth)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
epochs = 300
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device:{device}")


# Get sample input dimensions for model initialization
sample_batch = next(iter(dataloader))
b, f, h, w, c = sample_batch[0].shape
model = Model(c, f, h, w).to(device)
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
grad_accumulation_steps = 4

n_params = sum(p.numel() for p in model.parameters())


for epoch in range(epochs):
        model.train()
        print(f"Starting epoch {epoch+1}/{epochs}")
        batch_count = 0
        epoch_loss = 0
        valid_batches = 0
        optimizer.zero_grad()
        for data, labels in dataloader:
            if data is None or len(data) == 0:
                continue

            batch_count += 1

            try:

                data = data.permute(0, 4, 1, 2, 3).contiguous().to(device)

                #for ctc loss
                target_lengths = torch.tensor([len(yi) for yi in labels], device=device)
                input_lengths = torch.full((len(labels),), 75, device=device)

                # stack labels for ctc loss
                stacked_labels = torch.cat([label.to(device) for label in labels])


                log_probs = model(data)
                log_probs = log_probs.permute(1, 0, 2)


                loss = ctc_loss(log_probs, stacked_labels, input_lengths, target_lengths)


                loss = loss / grad_accumulation_steps


                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Batch {batch_count}: nan or inf loss detected, skipping")
                    # Clear any gradients from this batch
                    optimizer.zero_grad()
                    continue


                loss.backward()


                if batch_count % grad_accumulation_steps == 0:

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) #to avoid exploding grads ( i was getting a lot of NaN and this helped)
                    optimizer.step()
                    optimizer.zero_grad()
                    clear_gpu_cache()


                epoch_loss += loss.item() * grad_accumulation_steps
                valid_batches += 1


                if batch_count % 5 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_count}, Loss: {loss.item() * grad_accumulation_steps:.5f}")


                del data, log_probs, loss
                clear_gpu_cache()

            except Exception as e:
                print(f"Error in batch {batch_count}: {e}")
                clear_gpu_cache()

        if valid_batches > 0:
           avg_loss  = epoch_loss/valid_batches
           scheduler.step(avg_loss)
           print(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}, Valid batches: {valid_batches}/{batch_count}")

            # Save model checkpoint
           torch.save({
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'loss':avg_loss,


           },f'model_checkpoint_epoch_{epoch+1}.pt')
           clear_gpu_cache()
        else:
            print(f"Epoch {epoch+1} had no valid batches")
else:
    print("No valid data found")
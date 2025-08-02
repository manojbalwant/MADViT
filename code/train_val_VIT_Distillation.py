import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import timm
from torch.nn import init
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from data_utils_norm import DataLoader, compute_mean_and_std
from utils import annot_peaks
import cv2

# --- Configuration Management ---
class Config:
    def __init__(self):
        self.data_path = 'dataset/Drone-Anomaly/Bike Roundabout/' #'dataset/UIT-ADrone/'   #'dataset/shanghaitech/'   
        self.save_path = 'experiments/Bike Roundabout/' #'experiments/UIT-ADrone/' #'experiments/shanghaitech/'  
        self.valid_file = 'b_s1_te_01'  #'DJI_0146'  #'01_0025'    
        self.lr = 5e-4 #1e-3#1e-3
        self.wd = 5e-2#1e-2#5e-2
        self.image_size = 224
        self.patch_size = 16
        self.batch_size = 64
        self.joint_epochs = 15
        self.num_workers = 8
        self.num_frames = 1
        self.frame_step = 1  # Adjustable for temporal sampling
        self.num_pred = 0
        self.train = 1
        self.loss_type= 'token_loss'
        
def print_config(config):
    # Print config summary
    print("Configuration Summary:")
    for key, value in config.__dict__.items():
        print(f"{key}: {value}")

# --- Multi-layer Distillation Loss ---
def multi_layer_distillation_loss(teacher_feats, teacher_token, student_feats, student_token):
    # Feature loss layer-wise
    feature_loss = 0
    for t_feat, s_feat in zip(teacher_feats, student_feats):
        feature_loss += 1 - F.cosine_similarity(t_feat[:, 2:].flatten(1), s_feat[:, 2:].flatten(1)).mean()
    feature_loss /= len(teacher_feats)
    
    # Token loss (unchanged)
    temperature = 4.0
    token_loss = 0.5 * (
        F.kl_div(F.log_softmax(student_token[0] / temperature, dim=-1), 
                 F.softmax(teacher_token[0] / temperature, dim=-1), reduction='batchmean') +
        F.kl_div(F.log_softmax(student_token[1] / temperature, dim=-1), 
                 F.softmax(teacher_token[1] / temperature, dim=-1), reduction='batchmean')
    ) * (temperature ** 2)
    
    return token_loss

def cal_anomaly_map(fs_list, ft_list, img, mean, std, out_size=224, amap_mode='max'):
    """
    Calculate and plot the aggregated anomaly map given lists of features,
    using MSE to compute the anomaly score per patch for robust anomaly localization.
    
    Args:
        fs_list (list of torch.Tensor): List of student features with shape (B, num_blocks, patches, latent_dim).
        ft_list (list of torch.Tensor): List of teacher features with shape (B, num_blocks, patches, latent_dim).
        img (torch.Tensor): Input image tensor with shape (B, C, H, W).
        mean (list): Mean values for image normalization.
        std (list): Standard deviation values for image normalization.
        out_size (int): Output size for the anomaly map (default: 224).
        amap_mode (str): Aggregation mode for anomaly maps; 'mul' for multiplication, 'max' for maximum, else addition.
    
    Returns:
        anomaly_map (np.ndarray): Aggregated anomaly map of shape (out_size, out_size).
        a_map_list (list of np.ndarray): List of individual anomaly maps per layer.
    """
    import torch.nn.functional as F
    import numpy as np
    from scipy.ndimage import gaussian_filter
    import matplotlib.pyplot as plt

    # Initialize aggregated anomaly map based on aggregation mode
    if amap_mode == 'mul':
        anomaly_map = np.ones((out_size, out_size))
    elif amap_mode == 'max':
        anomaly_map = np.zeros((out_size, out_size))
    else:
        anomaly_map = np.zeros((out_size, out_size))
    a_map_list = []

    # Iterate over feature pairs from student and teacher models
    for fs, ft in zip(fs_list, ft_list):
        # Compute anomaly score using MSE, excluding cls and dist tokens
        a_map = 1 - F.cosine_similarity(fs[:, 2:], ft[:, 2:], dim=-1)#F.mse_loss(fs[:, 2:], ft[:, 2:], reduction='none').mean(dim=-1)
        
        # Determine grid size from the number of patches
        B, num_patches = a_map.shape
        grid_size = int(np.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            raise ValueError("Number of patches must be a perfect square.")
        
        # Reshape anomaly scores to a 2D map
        a_map = a_map.view(B, 1, grid_size, grid_size)
        
        # Upsample to the desired output size using bilinear interpolation
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        
        # Convert to NumPy for aggregation and visualization
        a_map_np = a_map[0, 0, :, :].cpu().detach().numpy()
        a_map_list.append(a_map_np)
        
        # Aggregate anomaly maps across layers
        if amap_mode == 'mul':
            anomaly_map *= a_map_np
        elif amap_mode == 'max':
            anomaly_map = np.maximum(anomaly_map, a_map_np)
        else:
            anomaly_map += a_map_np

    # Apply Gaussian smoothing to reduce noise
    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
    
    # Normalize the anomaly map to [0, 1]
    def min_max_norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
    norm_map = min_max_norm(anomaly_map)
    
    # Denormalize the input image for visualization
    img_np = img.detach().cpu().squeeze(0) * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
    img_np = img_np.clamp(0, 1).permute(1, 2, 0).numpy()

    # Visualize the input image and overlay the anomaly map
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title("Input Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_np)
    plt.imshow(norm_map, cmap='jet', alpha=0.5)
    plt.title("Anomaly Map Overlay")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

    return anomaly_map, a_map_list
    
# --- Batch Plotting ---
def plot_one_batch(batch_image, mean, std):
    images = batch_image['standard'][0]
    mean_tensor = torch.tensor(mean).view(1, 3, 1, 1)
    std_tensor  = torch.tensor(std).view(1, 3, 1, 1)
    images = images * std_tensor + mean_tensor
    images = images.permute(0, 2, 3, 1).numpy()
    
    batch_size = images.shape[0]
    plt.figure(figsize=(12, 6))
    for i in range(batch_size):
        plt.subplot(1, batch_size, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()

# --- Weight Initialization ---
def kaiming_init(module):
    """
    Applies Kaiming initialization to linear and convolutional layers,
    and standard initialization to layer normalization in the ViT model.
    
    Args:
        module: A PyTorch module from the model.
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        # Apply Kaiming normal initialization to weights
        init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        # Initialize biases to 0 if they exist
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        # Initialize LayerNorm weights to 1 and biases to 0
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)

# --- Teacher Model ---
class Teacher(nn.Module):
    """
    Teacher network based on a ViT backbone.
    This model is pretrained and frozen. It provides intermediate representations (skip connections)
    and token embeddings for multi-layer distillation.
    """
    def __init__(self, model_name='deit_base_distilled_patch16_224', pretrained=True, latent_dim=192):
        super(Teacher, self).__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        self.vit.head = nn.Identity()

        # Create bottleneck modules for selected transformer layers.
        self.bottlenecks = nn.ModuleList([
            nn.Sequential(nn.Linear(self.vit.embed_dim, latent_dim), nn.ReLU(inplace=True))
            for _ in range(len(self.vit.blocks))
        ])
        self.cls_bottleneck = nn.Sequential(nn.Linear(self.vit.embed_dim, latent_dim), nn.ReLU(inplace=True))
        self.dist_bottleneck = nn.Sequential(nn.Linear(self.vit.embed_dim, latent_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        """
        Forward pass for the teacher network.
        
        Args:
            x (Tensor): Input image batch of shape (B, C, H, W)
        
        Returns:
            compressed_skips (list[Tensor]): List of compressed skip connection features.
            tokens (tuple[Tensor, Tensor]): (cls_token_out, dist_token_out)
        """
        skip_connections = []
        
        # Patch embedding.
        x = self.vit.patch_embed(x)  # Shape: (B, N, D)
        
        # Prepare classification and distillation tokens.
        B, N, D = x.shape
        cls_token = self.vit.cls_token.expand(B, -1, -1)  # (B, 1, D)
        dist_token = self.vit.dist_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_token, dist_token, x), dim=1)  # (B, N+2, D)
        
        # Add positional embeddings and apply dropout.
        x = x + self.vit.pos_embed
        #x = self.vit.pos_drop(x)

        # Pass through transformer blocks, collecting features at selected layers.
        for i, block in enumerate(self.vit.blocks):
            x = block(x)
            #if i in [3, 6, 9, 11]:  # Collect skip connections.
            skip_connections.append(x)
        
        # Compress skip connection features.
        compressed_skips = [bottleneck(skip) for skip, bottleneck in zip(skip_connections, self.bottlenecks)]

        # Normalize final block output before extracting tokens.
        x = self.vit.norm(x)
        
        # Extract and compress final cls and distillation tokens.
        cls_token_out = self.cls_bottleneck(x[:, 0])  # (B, latent_dim)
        dist_token_out = self.dist_bottleneck(x[:, 1])  # (B, latent_dim)
        
        return compressed_skips, (cls_token_out, dist_token_out)


# --- Student Model ---
class Student(nn.Module):
    """
    Student network based on a smaller ViT backbone.
    It is randomly initialized and trained to mimic the teacher’s intermediate representations.
    """
    def __init__(self, model_name='deit_tiny_distilled_patch16_224', pretrained=False, embed_dim=192):
        super(Student, self).__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, embed_dim=embed_dim)
        self.vit.head = nn.Identity()
        
        # If training from scratch, apply custom initialization
        if not pretrained:
            self.vit.apply(kaiming_init)

    def forward(self, x):
        """
        Forward pass for the student network.
        
        Args:
            x (Tensor): Input image batch of shape (B, C, H, W)
            
        Returns:
            skip_connections (list[Tensor]): List of skip connection features.
            tokens (tuple[Tensor, Tensor]): (cls_token, dist_token)
        """
        skip_connections = []
        
        # Patch embedding.
        x = self.vit.patch_embed(x)  # Shape: (B, N, D)
        
        # Prepare classification and distillation tokens.
        B, N, D = x.shape
        cls_token = self.vit.cls_token.expand(B, -1, -1)  # (B, 1, D)
        dist_token = self.vit.dist_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_token, dist_token, x), dim=1)  # (B, N+2, D)
        
        # Add positional embeddings and dropout.
        x = x + self.vit.pos_embed
        #x = self.vit.pos_drop(x)

        # Pass through transformer blocks, collecting features at selected layers.
        for i, block in enumerate(self.vit.blocks):
            x = block(x)
            #if i in [3, 6, 9, 11]:
            skip_connections.append(x)

        # Normalize final block output.
        x = self.vit.norm(x)
        # Extract tokens.
        cls_token = x[:, 0]
        dist_token = x[:, 1]
        return skip_connections, (cls_token, dist_token)
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Training Function ---
def train_epoch(teacher, student, loader, optimizer, scheduler, scaler, config, mean, std, device):
    teacher.eval()
    student.train()
    epoch_loss = 0.0
    for batch in tqdm(loader, desc="Training"):
        inputs = batch["standard"].to(device)  # (B, T, C, H, W)
        inputs = inputs[:, 0].float()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                teacher_emb, teach_cls_token = teacher(inputs)
            student_emb, stu_cls_token = student(inputs)
            loss = multi_layer_distillation_loss(teacher_emb, teach_cls_token, student_emb, stu_cls_token)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

# --- Validation Function ---
def validate(teacher, student, loader, config, device):
    student.eval()
    teacher.eval()
    losses = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch["standard"].to(device)
            last_prev_frame = inputs[:, 0].float()
            #target_frame = inputs[:, config.num_frames].float()
        
            t_emb, teacher_token = teacher(last_prev_frame)
            s_emb, student_token = student(last_prev_frame)
            loss = multi_layer_distillation_loss(t_emb, teacher_token, s_emb, student_token)
            losses.append(loss.item())
    
    losses = np.array(losses)
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    true_labels = np.load(os.path.join(config.data_path, 'test/test_frame_mask/', config.valid_file + '.npy'))
    auc = roc_auc_score(true_labels[:len(losses)], losses)
    print(f"Validation AUC: {auc:.4f}, Loss: {np.mean(losses):.4f}")
    return np.mean(losses), auc

# --- Evaluation Function ---
def evaluate(teacher, student, config, mean, std, device=torch.device('cpu')):
    train_folder = os.path.join(config.data_path, 'train/frames')
    test_path =  os.path.join(config.data_path, 'test/')
    student.eval()
    teacher.eval()
    path_scenes = sorted(glob.glob(os.path.join(test_path, 'frames/*')))
    label_path = os.path.join(test_path, 'test_frame_mask/')
    
    all_losses = []
    all_labels = []

    for idx_video, path_scene in enumerate(path_scenes):
        print(f'Evaluating Video {idx_video + 1}/{len(path_scenes)}: {os.path.basename(path_scene)}')
        
        test_dataset = DataLoader(
            path_scene,
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]),
            resize_height=config.image_size,
            resize_width=config.image_size,
            time_step=config.num_frames,
        )
        test_loader = data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True
        )

        np_label = np.load(os.path.join(label_path, f"{os.path.basename(path_scene)}.npy"), allow_pickle=True)
        video_losses = []
        print(f"Scene: {os.path.basename(path_scene)} | Frames: {len(test_dataset)} | Labels: {np_label.size}")

        with torch.no_grad(), tqdm(desc=f'Evaluating {os.path.basename(path_scene)}', total=len(test_loader)) as pbar:
            for batch_data in test_loader:
                batch_data = batch_data['standard'].to(device, non_blocking=True)
                last_prev_frame = batch_data[:, 0].float()
                #target_frame = batch_data[:, config.num_frames].float()
                # Temporarily enable gradients for localization
                #with torch.enable_grad():
                    #gradients_localization(student, teacher, last_prev_frame, mean, std, plot=True)
                
                t_emb, teacher_token = teacher(last_prev_frame)
                s_emb, student_token = student(last_prev_frame)
                loss = multi_layer_distillation_loss(t_emb, teacher_token, s_emb, student_token)
                video_losses.append(loss.item())
                all_losses.append(loss.item())
                pbar.update()

        np.save(os.path.join(config.save_path, f"{os.path.basename(path_scene)}.npy"), np.array(video_losses))
        all_labels.append(np_label[-len(video_losses):])

    all_losses = np.array(all_losses)
    all_losses = (all_losses - all_losses.min()) / (all_losses.max() - all_losses.min())
    all_labels = np.concatenate(all_labels)
    frame_auc = roc_auc_score(y_true=all_labels, y_score=all_losses)
    mean_loss = np.mean(all_losses)

    fpr, tpr, thresholds = roc_curve(y_true=all_labels, y_score=all_losses)
    roc_df = pd.DataFrame({'tf': tpr - (1 - fpr), 'thresholds': thresholds})
    optimal_threshold = roc_df.iloc[(roc_df.tf).abs().argmin()]['thresholds']

    print(f"\nEvaluation Complete: AUC: {frame_auc:.4f}, Mean Loss: {mean_loss:.4f}, Optimal Threshold: {optimal_threshold}")
    return frame_auc
    
# --- Evaluation Function ---
def evaluate_with_plot(teacher, student, config, mean, std, device=torch.device('cpu')):
    train_folder = os.path.join(config.data_path, 'train/frames')
    test_path =  os.path.join(config.data_path, 'test/')
    student.eval()
    teacher.eval()
    path_scenes = sorted(glob.glob(os.path.join(test_path, 'frames/*')))
    label_path = os.path.join(test_path, 'test_frame_mask/')
    
    all_losses = []
    all_labels = []

    for idx_video, path_scene in enumerate(path_scenes):
        print(f'Evaluating Video {idx_video + 1}/{len(path_scenes)}: {os.path.basename(path_scene)}')
        
        test_dataset = DataLoader(
            path_scene,
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]),
            resize_height=config.image_size,
            resize_width=config.image_size,
            time_step=config.num_frames,
        )
        test_loader = data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True
        )

        np_label = np.load(os.path.join(label_path, f"{os.path.basename(path_scene)}.npy"), allow_pickle=True)
        video_losses = []
        print(f"Scene: {os.path.basename(path_scene)} | Frames: {len(test_dataset)} | Labels: {np_label.size}")

        with torch.no_grad(), tqdm(desc=f'Evaluating {os.path.basename(path_scene)}', total=len(test_loader)) as pbar:
            for batch_data in test_loader:
                batch_data = batch_data['standard'].to(device, non_blocking=True)
                last_prev_frame = batch_data[:, 0].float()
                
                t_emb, teacher_token = teacher(last_prev_frame)
                s_emb, student_token = student(last_prev_frame)
                
                #cal_anomaly_map(s_emb, t_emb, last_prev_frame, mean,std)
                loss = multi_layer_distillation_loss(t_emb, teacher_token, s_emb, student_token)
                video_losses.append(loss.item())
                all_losses.append(loss.item())
                pbar.update()

        np.save(os.path.join(config.save_path, f"{os.path.basename(path_scene)}.npy"), np.array(video_losses))
        all_labels.append(np_label[-len(video_losses):])

    all_losses = np.array(all_losses)
    all_losses = (all_losses - all_losses.min()) / (all_losses.max() - all_losses.min())
    all_labels = np.concatenate(all_labels)
    frame_auc = roc_auc_score(y_true=all_labels, y_score=all_losses)
    mean_loss = np.mean(all_losses)
    
    # Normalize losses for plotting.
    losses = all_losses  # use the input array directly
    losses_min, losses_max = np.min(losses), np.max(losses)
    norm_losses = (losses - losses_min) / (losses_max - losses_min)
    x_indices = np.arange(norm_losses.shape[0])
    
    # Plot predicted and actual scores on the same axis.
    fig, ax = plt.subplots()
    ax.plot(x_indices, norm_losses, label='predicted')
    ax.plot(all_labels, label='actual')
    #annot_peaks(x_indices, norm_losses, ax, peak_distance=2000, y_position_modifier=1)
    fig.subplots_adjust(bottom=0.6)
    ax.legend(prop={"size": 6}, loc='lower right')
    plt.show()
    plt.savefig(os.path.join(config.save_path, 'score_plot.png')) 
    
    # Compute ROC curve metrics.
    fpr, tpr, thresholds_roc = roc_curve(y_true=all_labels, y_score=losses)
    fnr = 1 - tpr
    
    # Compute the threshold corresponding to FPR <= 0.3.
    valid_indices = np.where(fpr <= 0.3)[0]
    threshold_fp = thresholds_roc[valid_indices[-1]] if valid_indices.size else thresholds_roc[0]
    
    # Find the Equal Error Rate (EER): the threshold where |FPR - FNR| is minimized.
    idx_eer = np.nanargmin(np.abs(fnr - fpr))
    eer_threshold = thresholds_roc[idx_eer]
    eer = fpr[idx_eer]
    
    # Compute the optimal threshold where tpr is closest to (1 - fpr).
    optimal_idx = np.argmin(np.abs(tpr - (1 - fpr)))
    optimal_threshold = thresholds_roc[optimal_idx]
    print('Optimal threshold value:', optimal_threshold)
    
    # Save the optimal threshold.
    filename_net = os.path.join(config.save_path, 'checkpoints', 'optimal_thres.pth')
    os.makedirs(os.path.dirname(filename_net), exist_ok=True)
    torch.save(optimal_threshold, filename_net)
    
    frame_auc = auc(fpr, tpr)
    print("Evaluation results: AUC@1: {:.4f} EER@1: {:.4f} - Mean loss: {:.2f}".format(frame_auc, eer, mean_loss))
    
    return frame_auc

# --- Main Function ---
def main():
    set_seed(42)
    config = Config()
    print_config(config)
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    
    teacher = Teacher('deit_base_distilled_patch16_224', pretrained=True)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    teacher = nn.DataParallel(teacher, device_ids=[7]).to(device)
    
    student = Student('deit_tiny_distilled_patch16_224', pretrained=False)
    student.train()
    student = nn.DataParallel(student, device_ids=[7]).to(device)
    
    train_folder = os.path.join(config.data_path, 'train/frames')
    mean, std = compute_mean_and_std(train_folder, config.image_size, config.image_size, device=device, batch_size=config.batch_size)
    print('mean, std, lr, wd:', mean, std, config.lr, config.wd)
    
    if config.train:
        train_dataset = DataLoader(
            train_folder,
            transforms.Compose([
                #transforms.RandomApply([transforms.RandomResizedCrop(224)], p=0.5),
                #transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)], p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomApply([transforms.RandomRotation(15)], p=0.5),
                #transforms.RandomApply([transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3))], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]),
            resize_height=config.image_size,
            resize_width=config.image_size,
            time_step=config.num_frames
        )
        train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                       num_workers=config.num_workers, pin_memory=True, drop_last=True)
        
        valid_folder = os.path.join(config.data_path, 'test/frames/', config.valid_file)
        valid_dataset = DataLoader(
            valid_folder,
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]),
            resize_height=config.image_size,
            resize_width=config.image_size,
            time_step=config.num_frames
        )
        valid_loader = data.DataLoader(valid_dataset, batch_size=1, shuffle=False,
                                       num_workers=config.num_workers, pin_memory=True, drop_last=True)
        
        plot_one_batch(next(iter(train_loader)), mean, std)
        
        optimizer = torch.optim.AdamW(student.parameters(), lr=config.lr, weight_decay=config.wd)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.lr, steps_per_epoch=len(train_loader), epochs=config.joint_epochs
        )
        scaler = torch.cuda.amp.GradScaler()
        
        print("\n===== Stage 2: Joint Training =====")
        best_auc = 0
        for epoch in range(config.joint_epochs):
            train_loss = train_epoch(teacher, student, train_loader, optimizer, scheduler, scaler, config, mean, std, device)
            val_loss, val_auc = validate(teacher, student, valid_loader, config, device)
            print(f"Epoch {epoch+1}/{config.joint_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        torch.save({'state_dict': student.state_dict(), 'mean_std': {'mean': mean, 'std': std}},
                           os.path.join(config.save_path, 'checkpoints', 'best_t.pth'))
        best_test_auc = evaluate(teacher, student, config, mean, std, device)
    else:
        print('Testing ...')
        path_ckpt = os.path.join(config.save_path, 'checkpoints', 'best_t.pth')
        checkpoint = torch.load(path_ckpt, map_location=device)
        student.load_state_dict(checkpoint['state_dict'])
        evaluate_with_plot(teacher, student, config, mean, std, device)
    
    return 0

if __name__ == '__main__':
    main()

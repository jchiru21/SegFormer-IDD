import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from dataset import IDDSegmentationDataset
from utils import compute_miou
from tqdm import tqdm
import os

# ----------------- SETTINGS -----------------
DATA_ROOT = "/content/drive/MyDrive/Datasets/IDD/IDD_Segmentation"
NUM_CLASSES = 19
BATCH_SIZE = 2
EPOCHS = 2
LR = 3e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IGNORE_INDEX = 255

# ----------------- DATASET -----------------
# Original label IDs in your mask
ORIGINAL_LABEL_IDS = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
label_remap = {orig:i for i, orig in enumerate(ORIGINAL_LABEL_IDS)}

transform = transforms.ToTensor()

train_dataset = IDDSegmentationDataset(
    DATA_ROOT + "/leftImg8bit/train",
    DATA_ROOT + "/gtFine_masks/train",
    transform=transform,
    label_remap=label_remap
)
val_dataset = IDDSegmentationDataset(
    DATA_ROOT + "/leftImg8bit/val",
    DATA_ROOT + "/gtFine_masks/val",
    transform=transform,
    label_remap=label_remap
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ----------------- MODEL -----------------
model_name = "nvidia/segformer-b2-finetuned-ade-512-512"
feature_extractor = SegformerImageProcessor(do_resize=True, size=(512,512), do_normalize=True)

model = SegformerForSemanticSegmentation.from_pretrained(
    model_name,
    num_labels=NUM_CLASSES,
    id2label={i:str(i) for i in range(NUM_CLASSES)},
    label2id={str(i):i for i in range(NUM_CLASSES)},
    ignore_mismatched_sizes=True
)
model.to(DEVICE)

# ----------------- OPTIMIZER -----------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler(enabled=True)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

# ----------------- TRAIN LOOP -----------------
save_dir = os.path.join(DATA_ROOT, "segformer_checkpoints")
os.makedirs(save_dir, exist_ok=True)
BEST_MIOU = 0.0

for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Train E{epoch}"):
        pv = batch["pixel_values"].to(DEVICE)
        lbl = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            out = model(pixel_values=pv, labels=lbl)
            loss = out.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch} train loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    miou_total = 0.0
    with torch.no_grad():
        for batch in val_loader:
            pv = batch["pixel_values"].to(DEVICE)
            lbl = batch["labels"].to(DEVICE)
            logits = model(pixel_values=pv).logits
            preds = logits.argmax(dim=1)
            miou_total += compute_miou(preds, lbl, NUM_CLASSES)

    val_miou = miou_total / len(val_loader)
    print(f"Epoch {epoch} Validation mIoU: {val_miou:.4f}")

    # Save checkpoint
    ckpt_path = os.path.join(save_dir, f"epoch{epoch}_miou{val_miou:.4f}.pt")
    torch.save(model.state_dict(), ckpt_path)
    if val_miou > BEST_MIOU:
        BEST_MIOU = val_miou
        best_ckpt = os.path.join(save_dir, "best_model.pt")
        torch.save(model.state_dict(), best_ckpt)
        print("New best mIoU saved:", BEST_MIOU)

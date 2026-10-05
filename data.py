"""What every step shares: the split, the classes, the backbone, labels, frame datasets and features.

Machine-specific paths and the GPU are not fixed here: every script takes them as command-line options.
"""
import os
import warnings

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # torch GPU indices = nvidia-smi indices

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

warnings.filterwarnings("ignore", message="xFormers is not available")

# Split by video (= patient). Test = the standard Cholec80 test set, used once for the final evaluation.
TRAIN = list(range(1, 33))
VAL = list(range(33, 41))
TEST = list(range(41, 81))

PHASES = ["Preparation", "CalotTriangleDissection", "ClippingCutting", "GallbladderDissection",
          "GallbladderPackaging", "CleaningCoagulation", "GallbladderRetraction"]
TOOLS = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]

# DINOv2 ViT-B/14 with 4 registers; code pinned to one GitHub commit (torch.hub downloads it and the weights)
DINOV2_REPO = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_MODEL = "dinov2_vitb14_reg"

# Whole frame resized to 224x392 = 16 x 28 patches (no crop, so tools at the image edges stay in view)
INPUT_SIZE = (224, 392)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
EVAL_TF = T.Compose([T.Resize(INPUT_SIZE, interpolation=T.InterpolationMode.BICUBIC),
                     T.ToTensor(), T.Normalize(MEAN, STD)]) #resize to 224x392 and normalize the way dino expects


def video_labels(frames, annotations, v):
    
    frame_ids = sorted(int(p.stem) for p in (frames / f"video{v:02d}").glob("*.jpg")) #List the frame images that extract_frames.py saved
    phase_of, tools_of = {}, {}


    #read the two label files for video v and fill two dictionaries that map frame number → label (both tool and phase):
    for line in (annotations / "phase_annotations" / f"video{v:02d}-phase.txt").read_text().splitlines()[1:]:
        frame, phase = line.split("\t")
        phase_of[int(frame)] = PHASES.index(phase)
    
    
    for line in (annotations / "tool_annotations" / f"video{v:02d}-tool.txt").read_text().splitlines()[1:]:
        frame, *present = line.split("\t")
        tools_of[int(frame)] = [int(x) for x in present]
    
    
    return {"frame": torch.tensor(frame_ids),
            "phase": torch.tensor([phase_of[f] for f in frame_ids]),
            "tools": torch.tensor([tools_of.get(f, [-1] * len(TOOLS)) for f in frame_ids], dtype=torch.int8)}

#Pytorch dataset
#All frames of the given videos in time order. labels=False: images only (self-supervised pretraining);
class Frames(Dataset):

   #builds a list of what each frame is and where it lives.
    def __init__(self, frames, annotations, videos, transform=EVAL_TF, labels=True):
        self.items, self.transform, self.labels = [], transform, labels
        for v in videos:
            lab = video_labels(frames, annotations, v)
            for f, p, t in zip(lab["frame"].tolist(), lab["phase"].tolist(), lab["tools"].tolist()):
                self.items.append((frames / f"video{v:02d}" / f"{f:06d}.jpg", p, t, v))
    #return number of frames
    def __len__(self):
        return len(self.items)
    
    #return a single frame (batching happens inside dataloader)
    def __getitem__(self, i):
        path, phase, tools, v = self.items[i]
        img = self.transform(Image.open(path).convert("RGB"))
        return (img, phase, torch.tensor(tools, dtype=torch.float), v) if self.labels else img


# Used for:
#load_backbone("cuda:0") → original DINOv2 (the "baseline")
#load_backbone("cuda:0", "stage1.pth")
def load_backbone(device, checkpoint=None):
   
    model = torch.hub.load(DINOV2_REPO, DINOV2_MODEL, trust_repo=True) #download backbone from github

    #if a checkpoint is given, it replaces those original weights with ones from an earlier training
    if checkpoint is not None:
        teacher = torch.load(checkpoint, map_location="cpu")["teacher"]
        model.load_state_dict({k.removeprefix("backbone."): v for k, v in teacher.items() if k.startswith("backbone.")}) #remove the prefix "backbone" from name
    model.requires_grad_(False)
    return model.to(device).eval()


# given a single frame, create a one CLS token (size=768) for the whole image, and one for each patch where each patch is 14x14 pixel patch
def cls_and_patch_mean(backbone, x):
    out = backbone.forward_features(x)
    return out["x_norm_clstoken"].float(), out["x_norm_patchtokens"].float().mean(1)


#Run every frame of the chosen videos through DINOv2 once, 
# collect the two 768-number vectors per frame, and save them per video 
# together with the labels. Later steps then train on these saved vectors 
# instead of re-running the slow model (Dino).
@torch.no_grad()
def extract_features(backbone, frames, annotations, videos, device, out_dir=None, workers=32, per_video=False):
    """Features of every frame of the given videos (bf16 autocast, no augmentation), in one pass.
    Saved per video as {frame, phase, tools, cls, patch_mean} when out_dir is given; also returned.
    In bf16 the exact values depend on how frames are batched: the frozen backbones' features were extracted for all
    80 videos in one pass, the LoRA backbones' video by video (per_video=True); this reproduces them exactly."""
    if per_video:
        result = {}
        for v in videos:
            result |= extract_features(backbone, frames, annotations, [v], device, out_dir, workers)
        return result
    loader = DataLoader(Frames(frames, annotations, videos), batch_size=256, num_workers=workers, pin_memory=True)
    cls, pm = [], []
    for x, *_ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            c, p = cls_and_patch_mean(backbone, x.to(device, non_blocking=True))
        cls.append(c.cpu()), pm.append(p.cpu())
    cls, pm = torch.cat(cls), torch.cat(pm)
    result, start = {}, 0
    for v in videos:
        lab = video_labels(frames, annotations, v)
        n = len(lab["frame"])
        result[v] = {**lab, "cls": cls[start:start + n].clone(), "patch_mean": pm[start:start + n].clone()}
        start += n
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(result[v], out_dir / f"video{v:02d}.pt")
    return result


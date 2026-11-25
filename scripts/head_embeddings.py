"""Head-crop features from the L2CS gaze network (Gaze360-trained ResNet50).

Same boxes/frames as crop_embeddings.py. For each student box the head region
(top ~45% of the box, centred) is cropped and passed through L2CS-Net:
2048-D pooled features + predicted gaze yaw/pitch (degrees).
Output: data/processed/head_embeddings.npz (aligned row-for-row with crop_embeddings.npz).
"""
from __future__ import annotations
import sys
from pathlib import Path
import cv2, numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from attention_pipeline.models.l2cs_net import L2CSNet
from crop_embeddings import VIDEOS

def main():
    d = np.load(ROOT / "data/processed/crop_embeddings.npz", allow_pickle=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = L2CSNet().to(dev).eval(); net.load(ROOT / "checkpoints/l2cs_finetuned.pt", map_location=dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    idx_bins = torch.arange(90, device=dev, dtype=torch.float32)
    feats = np.zeros((len(d["label"]), 2048), np.float32); gaze = np.zeros((len(d["label"]), 2), np.float32)
    vfile = {v: f for _, v, f, _ in VIDEOS}
    for v in np.unique(d["video"]):
        rows = np.where(d["video"] == v)[0]; byf = {}
        for r in rows: byf.setdefault(int(d["frame"][r]), []).append(r)
        cap = cv2.VideoCapture(str(ROOT / vfile[v])); f = 0
        while True:
            ok, fr = cap.read()
            if not ok: break
            if f in byf:
                crops, rr = [], []
                for r in byf[f]:
                    x1, y1, x2, y2 = d["box"][r]; w, h = x2 - x1, y2 - y1; side = max(w * 0.75, h * 0.45)
                    cx, cy = (x1 + x2) / 2, y1 + side * 0.55
                    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2)); c, e = int(min(fr.shape[1], cx + side / 2)), int(min(fr.shape[0], cy + side / 2))
                    if c - a < 6 or e - b < 6: continue
                    crops.append(cv2.resize(cv2.cvtColor(fr[b:e, a:c], cv2.COLOR_BGR2RGB), (224, 224))); rr.append(r)
                if crops:
                    x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().div(255).to(dev); x = (x - mean) / std
                    with torch.no_grad():
                        z = net.maxpool(net.relu(net.bn1(net.conv1(x)))); z = net.layer4(net.layer3(net.layer2(net.layer1(z))))
                        z = torch.flatten(net.avgpool(z), 1)
                        yaw = (torch.softmax(net.fc_yaw_gaze(z), 1) * idx_bins).sum(1) * 4 - 180
                        pit = (torch.softmax(net.fc_pitch_gaze(z), 1) * idx_bins).sum(1) * 4 - 180
                    feats[rr] = z.cpu().numpy(); gaze[rr] = torch.stack([yaw, pit], 1).cpu().numpy()
            f += 1
        print(v, "done")
    np.savez_compressed(ROOT / "data/processed/head_embeddings.npz", l2cs=feats, gaze=gaze)

if __name__ == "__main__":
    main()

V-JEPA 2.1 ViT-B block5 checkpoint. This machine uses a symlink to the original 1.6 GB file. To make the repository self-contained, manually replace the symlink with an unchanged copy:

```bash
cp /home/hx/vjepa2-research/research/assets/model/vjepa2_1_vitb_dist_vitG_384.pt checkpoints/vjepa/vjepa2_1_vitb_dist_vitG_384.pt.tmp
mv -Tf checkpoints/vjepa/vjepa2_1_vitb_dist_vitG_384.pt.tmp checkpoints/vjepa/vjepa2_1_vitb_dist_vitG_384.pt
sha256sum checkpoints/vjepa/vjepa2_1_vitb_dist_vitG_384.pt
```

Expected SHA256: `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`.

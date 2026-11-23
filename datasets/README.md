# Datasets

The repository contains download links and preparation code. Dataset files remain with their original providers.

| Dataset | Download |
| --- | --- |
| MUUFL Gulfport | [University of Florida / GatorSense](https://github.com/GatorSense/MUUFLGulfport) |
| Houston 2013 | [IEEE GRSS Data Fusion Contest](https://www.grss-ieee.org/community/technical-committees/2013-ieee-grss-data-fusion-contest/) |
| Trento | [Public download mirror](https://github.com/tyust-dayu/Trento) |

Follow the providers' access conditions and cite the original datasets. Houston access is managed by its provider. Trento's linked repository is a mirror. MUUFL data and scene-label references are listed in the GatorSense repository.

## Prepare MATLAB files

Place the co-registered arrays under `datasets/raw/`:

```text
datasets/raw/
  muufl/
    MUF_HSI.mat
    MUF_LiDAR.mat
    MUF_gt.mat
  houston/
    Houston.mat
    Houston_LR.mat
    Houston_gt.mat
  trento/
    Italy_hsi.mat
    Italy_lidar.mat
    allgrd.mat
```

Expected HSI dimensions are `325 x 220 x 64` for MUUFL, `349 x 1905 x 144` for Houston, and `166 x 600 x 63` for Trento. LiDAR uses matching spatial dimensions: elevation followed by intensity for MUUFL, one elevation channel for Houston, and both supplied channels for Trento. Preserve their channel order. Ground-truth arrays use consecutive labels starting at 1 and 0 for unlabeled pixels.

```bash
python datasets/prepare.py --dataset muufl
python datasets/prepare.py --dataset houston
python datasets/prepare.py --dataset trento
```

Provider archives may use different filenames or MATLAB structures. Select the actual files and variables rather than renaming an archive:

```bash
python datasets/prepare.py --dataset muufl \
  --hsi-file /path/to/scene.mat --hsi-key scene.hsi \
  --lidar-file /path/to/scene.mat --lidar-key scene.lidar \
  --labels-file /path/to/labels.mat --labels-key labels
```

The variable names above illustrate the syntax; use the keys present in your files. If MUUFL elevation and intensity are separate arrays in the same file, pass both keys in that order, for example `--lidar-key scene.elevation,scene.intensity`. Numeric arrays in MATLAB v7.3 files are also supported. Preparation writes `datasets/prepared/<dataset>/arrays.npz`. For each seed, training selects 20 labeled pixels per class and testing uses the remaining labeled pixels. Normalization uses whole-scene image statistics without label information.

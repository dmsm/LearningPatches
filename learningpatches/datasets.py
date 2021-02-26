import os

import numpy as np
from PIL import Image
import torch as th
from torchvision import transforms


class SketchDataset(th.utils.data.Dataset):
    """Dataset of pairs of sketch images and target 3D geometry."""

    def __init__(self, args, jitter=True, val=False):
        """Initialize dataset with optional jittering of input images."""
        self.args = args
        self._jitter = transforms.Compose([
            transforms.Pad(20, fill=(255, 255, 255)),
            transforms.RandomRotation(15, resample=Image.BILINEAR),
            transforms.CenterCrop(args.im_size),
            transforms.RandomResizedCrop(args.im_size, (0.9, 1), (1, 1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor()
        ]) if jitter else transforms.ToTensor()

        self.root = args.data
        self.turbine_files = [f.strip() for f in
                              open(os.path.join(self.root, 'turbines.txt'),
                                   'r').readlines()] \
            if os.path.exists(os.path.join(self.root, 'turbines.txt')) else []

        if args.wheels:
            self.files = []
            for f in sorted([f for f in os.listdir(self.root)
                             if os.path.isdir(os.path.join(self.root, f))]):
                if np.load(os.path.join(self.root, f, 'wheels.npy')).sum() < \
                        int(args.n_samples*0.1232):
                    continue
                else:
                    self.files.append(f)
        else:
            self.files = sorted([f for f in os.listdir(
                self.root) if os.path.isdir(os.path.join(self.root, f))])
        cutoff = int(0.9*len(self.files))
        if val:
            self.files = self.files[cutoff:]
        else:
            self.files = self.files[:cutoff]

    def __repr__(self):
        return "SketchDataset | {} entries".format(len(self))

    def __len__(self):
        return len(self.files)

    def get(self, name):
        idx = self.files.index(name)
        return self[idx]

    def __getitem__(self, idx):
        model = self.files[idx]

        fnames = [f for f in os.listdir(os.path.join(
            self.root, model, 'pngs')) if f.endswith('.png')]

        ims = []
        if self.args.p2m:
            ims = self._jitter(
                Image.open(
                    os.path.join(self.root, model, 'pngs',
                                 np.random.choice(fnames))
                ).convert('RGB'))[None]

        else:
            views = set()
            for f in fnames:
                views.add(f.split('_')[0].replace('.png', ''))
            n_views = np.random.randint(
                1, min(self.args.n_views, len(views))+1)
            selected_views = np.random.choice(
                list(views), size=n_views, replace=False)
            fnames_per_view = [
                [f for f in fnames if f.startswith(v)] for v in selected_views]

            for fnames in fnames_per_view:
                f = np.random.choice(fnames)
                im = self._jitter(Image.open(os.path.join(
                    self.root, model, 'pngs', f)).convert('RGB'))
                ims.append(im)

            ims = th.stack(ims, dim=0)
            ims = th.cat([ims, ims[:1].expand(
                self.args.n_views-n_views, -1, -1, -1)], dim=0)

        samples = np.load(os.path.join(self.root, model,
                                       'samples.npy')).astype(np.float32)
        perm = np.random.permutation(samples.shape[0])
        points, normals = np.split(samples[perm], 2, axis=1)

        ratio = 0.
        if self.args.wheels:
            wheel_mask = np.load(os.path.join(
                self.root, model, 'wheels.npy'))[perm]
            ratio = wheel_mask.sum() / wheel_mask.size

            wheel_points = points[wheel_mask][:int(self.args.n_samples*0.1232)]
            wheel_normals = normals[wheel_mask][:int(
                self.args.n_samples*0.1232)]

            points = points[np.logical_not(wheel_mask)]
            normals = normals[np.logical_not(wheel_mask)]

            points = points[:self.args.n_samples]
            normals = normals[:self.args.n_samples]
        else:
            wheel_points = wheel_normals = np.array([])

            points = points[:self.args.n_samples]
            normals = normals[:self.args.n_samples]

        return {
            'fname': model,
            'ims': ims,
            'points': th.from_numpy(points),
            'normals': th.from_numpy(normals),
            'wheel_points': th.from_numpy(wheel_points),
            'wheel_normals': th.from_numpy(wheel_normals),
            'turbines': model in self.turbine_files,
            'ratio': th.tensor(ratio),
        }

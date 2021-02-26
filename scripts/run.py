import argparse
import ast
from collections import defaultdict
import os

import numpy as np
from PIL import Image
import torch as th
from torchvision.transforms.functional import to_tensor
import ttools

from learningpatches.models import ReconstructionModel
from learningpatches import utils

th.manual_seed(123)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = True
np.random.seed(123)

turbine_idxs = [x for x in range(76) if x not in [72, 73, 74, 75]]
no_turbine_idxs = [x for x in range(76) if x not in
                   [10, 12, 14, 17, 19, 20, 21, 24, 25, 27, 28, 29, 30, 31, 32,
                    33, 34, 66]]


def main(args):
    device = "cuda" if th.cuda.is_available() and args.cuda else "cpu"

    edge_data = [line.strip().split(' ') for line in open(
        os.path.join('templates', args.category, 'edges.txt'), 'r')]
    edge_data = [(int(a), int(b), int(c), int(d)) for a, b, c, d in edge_data]
    junction_order = [int(line.strip()) for line in open(
        os.path.join('templates', args.category, 'junction_order.txt'), 'r')]
    topology = ast.literal_eval(
        open(os.path.join('templates', args.category,
                          'topology.txt'), 'r').read())
    adjacencies = {}
    for i, l in enumerate(
            open(os.path.join('templates', args.category, 'adjacencies.txt'),
                 'r')):
        adj = {}
        for x in l.strip().split(','):
            if x != '':
                j, edge = x.strip().split(' edge ')
                adj[int(j)] = int(edge)
        adjacencies[i] = adj

    vertex_t = {}
    init_params = []
    junctions = {}
    vertex_idxs = np.zeros(
        [len(open(os.path.join(
            'templates', args.category, 'vertices.txt'), 'r').readlines()), 3],
        dtype=np.int64)
    processed_vertices = []
    for i, l in enumerate(
            open(os.path.join('templates', args.category, 'vertices.txt'),
                 'r')):
        value = l.strip().split(' ')
        if value[0] == 'Junction':
            v_type, v0, v1, v2, v3, t_init = value
            vertex_t[i] = len(init_params)
            init_params.append(utils.logit(float(t_init)))
            junctions[i] = (int(v0), int(v1), int(v2), int(v3))
        elif value[0] == 'RegularVertex':
            _, a, b, c = value
            vertex_idxs[i] = [len(init_params), len(
                init_params)+1, len(init_params)+2]
            init_params.extend([float(a), float(b), float(c)])
            processed_vertices.append(i)

    edge_data_ = defaultdict(list)
    processed_edges = []
    for i in junction_order:
        processed_vertices.append(i)
        for a, b, c, d in edge_data:
            if a in processed_vertices and \
                    d in processed_vertices and \
                    (a, b, c, d) not in processed_edges:
                edge_data_[i].append((a, b, c, d))
                processed_edges.append((a, b, c, d))
    edge_data = edge_data_

    face_idxs = np.empty([len(topology), 12])
    for i, patch in enumerate(topology):
        for j, k in enumerate(patch):
            face_idxs[i, j] = k
    face_idxs = th.from_numpy(face_idxs.astype(np.int64))

    init_params = th.tensor(init_params).squeeze()

    model = ReconstructionModel(len(init_params), init=init_params)
    model.to(device)
    model.eval()

    checkpointer = ttools.Checkpointer(f'models/{args.category}', model)
    extras, _ = checkpointer.load_latest()
    if extras is not None:
        print(f"Loaded checkpoint (epoch {extras['epoch']})")
    else:
        print("Unable to load checkpoint")

    with th.no_grad():
        im = to_tensor(Image.open(args.input).convert('RGB')).to(device)
        params = model(im[None, None])
        _, patches = utils.process_patches(
            params, vertex_idxs, face_idxs, edge_data,
            junctions, junction_order, vertex_t)
        if args.category == 'airplanes':
            if args.turbines:
                patches = patches.squeeze(0)[turbine_idxs]
            else:
                patches = patches.squeeze(0)[no_turbine_idxs]
        else:
            patches = patches.squeeze(0)
        utils.write_obj(args.output, patches)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("category", choices=[
                        'airplanes', 'guitars', 'bathtubs', 'knives', 'cars',
                        'bottles', 'guns', 'mugs'], default='airplanes')
    parser.add_argument("output", type=str)
    parser.add_argument("--cuda", dest='cuda', action='store_true')
    parser.add_argument("--no_cuda", dest='cuda', action='store_false')
    parser.add_argument("--turbines", dest='cuda', action='store_true')
    parser.add_argument("--no_turbines", dest='cuda', action='store_false')
    parser.set_defaults(cuda=True, turbines=True)
    args = parser.parse_args()
    main(args)

import ast
from collections import defaultdict
import datetime
import os
import logging

import numpy as np
import torch as th
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import ttools

from learningpatches import datasets
from learningpatches.interfaces import ReconstructionInterface
from learningpatches.models import ReconstructionModel
from learningpatches import utils


LOG = logging.getLogger(__name__)

th.manual_seed(123)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = True
np.random.seed(123)


def _worker_init_fn(worker_id):
    np.random.seed(worker_id)


def main(args):
    data = datasets.SketchDataset(args)
    dataloader = DataLoader(data, batch_size=args.bs,
                            num_workers=args.num_worker_threads,
                            worker_init_fn=_worker_init_fn, shuffle=True,
                            drop_last=True)
    LOG.info(data)

    val_data = datasets.SketchDataset(args, val=True)
    val_dataloader = DataLoader(val_data, batch_size=args.bs,
                                num_workers=args.num_worker_threads,
                                worker_init_fn=_worker_init_fn)

    edge_data = [line.strip().split(' ') for line in open(
        os.path.join(args.template_dir, 'edges.txt'), 'r')]
    edge_data = [(int(a), int(b), int(c), int(d)) for a, b, c, d in edge_data]
    junction_order = [int(line.strip()) for line in open(
        os.path.join(args.template_dir, 'junction_order.txt'), 'r')]
    topology = ast.literal_eval(
        open(os.path.join(args.template_dir, 'topology.txt'), 'r').read())
    adjacencies = {}
    for i, l in enumerate(
            open(os.path.join(args.template_dir, 'adjacencies.txt'), 'r')):
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
            args.template_dir, 'vertices.txt'), 'r').readlines()), 3],
        dtype=np.int64)
    processed_vertices = []
    for i, l in enumerate(
            open(os.path.join(args.template_dir, 'vertices.txt'), 'r')):
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
    init_patches = utils.process_patches(
        init_params[None], vertex_idxs, face_idxs, edge_data, junctions,
        junction_order, vertex_t)[1][0]
    st = th.empty(init_patches.shape[0], 1, 2).fill_(0.5).to(init_params)
    template_normals = utils.coons_normals(
        st[..., 0], st[..., 1], init_patches)

    model = ReconstructionModel(len(init_params), init=init_params)

    if args.symmetries:
        xs, ys = [], []
        for line in open(os.path.join(args.data, 'symmetries.txt'), 'r'):
            x, y = line.strip().split(' ')
            xs.append(int(x))
            ys.append(int(y))
        symmetries = (xs, ys)
    else:
        symmetries = None
    interface = ReconstructionInterface(
        model, args, vertex_idxs, face_idxs, junctions, edge_data, vertex_t,
        adjacencies, junction_order, template_normals, symmetries)
    checkpointer = ttools.Checkpointer(
        args.checkpoint_dir, model=model, optimizers=interface.optimizer)
    extras, meta = checkpointer.load_latest()

    keys = ['loss', 'chamferloss', 'normalsloss', 'collisionloss',
            'planarloss', 'templatenormalsloss', 'symmetryloss']

    writer = SummaryWriter(
        os.path.join(args.checkpoint_dir, 'summaries',
                     datetime.datetime.now().strftime('train-%m%d%y-%H%M%S')),
        flush_secs=1)
    val_writer = SummaryWriter(
        os.path.join(args.checkpoint_dir, 'summaries',
                     datetime.datetime.now().strftime('val-%m%d%y-%H%M%S')),
        flush_secs=1)

    trainer = ttools.Trainer(interface)
    trainer.add_callback(
        ttools.callbacks.TensorBoardLoggingCallback(keys=keys, writer=writer,
                                                    val_writer=val_writer,
                                                    frequency=5))
    trainer.add_callback(ttools.callbacks.ProgressBarCallback(keys=keys))
    trainer.add_callback(ttools.callbacks.CheckpointingCallback(
        checkpointer, max_files=1, max_epochs=2))

    starting_epoch = extras['epoch'] if extras is not None else None
    trainer.train(dataloader, num_epochs=args.num_epochs,
                  val_dataloader=val_dataloader, starting_epoch=starting_epoch)


if __name__ == '__main__':
    parser = ttools.BasicArgumentParser()
    parser.add_argument("--im_size", type=int, default=128)
    parser.add_argument("--template_dir", type=str)
    parser.add_argument("--w_normals", type=float, default=0.008)
    parser.add_argument("--w_collision", type=float, default=1e-5)
    parser.add_argument("--sigma_collision", type=float, default=1e-6)
    parser.add_argument("--w_planar", type=float, default=2)
    parser.add_argument("--w_templatenormals", type=float, default=1e-4)
    parser.add_argument("--w_symmetry", type=float, default=1)
    parser.add_argument("--n_samples", type=int, default=7000)
    parser.add_argument("--n_views", type=int, default=4)
    parser.add_argument('--seperate-turbines',
                        dest='seperate_turbines', action='store_true')
    parser.add_argument('--no-seperate-turbines',
                        dest='seperate_turbines', action='store_false')
    parser.add_argument('--wheels', dest='wheels', action='store_true')
    parser.add_argument('--no-wheels', dest='wheels', action='store_false')
    parser.add_argument('--p2m', dest='p2m', action='store_true')
    parser.add_argument('--no-p2m', dest='p2m', action='store_false')
    parser.add_argument('--symmetries', dest='symmetries', action='store_true')
    parser.add_argument('--no-symmetries',
                        dest='symmetries', action='store_false')
    parser.set_defaults(seperate_turbines=False, wheels=False, p2m=False,
                        symmetries=False, num_worker_threads=8, lr=1e-4, bs=8)
    args = parser.parse_args()
    ttools.set_logger(args.debug)
    main(args)

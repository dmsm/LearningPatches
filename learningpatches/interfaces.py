import itertools

import numpy as np
import torch as th
from ttools.training import ModelInterface

from . import utils


class ReconstructionInterface(ModelInterface):
    def __init__(self, model, args, vertex_idxs, face_idxs, junctions,
                 edge_data, vertex_t, adjacencies, junction_order,
                 template_normals, symmetries=None):
        self.model = model

        self.vertex_idxs = vertex_idxs
        self.face_idxs = face_idxs
        self.junctions = junctions
        self.edge_data = edge_data
        self.vertex_t = vertex_t
        self.junction_order = junction_order
        self.template_normals = template_normals[None]

        self.args = args

        self.n_samples_per_loop_side = int(
            np.ceil(np.sqrt(args.n_samples / face_idxs.shape[0])))

        self.optimizer = th.optim.Adam(self.model.parameters(), lr=args.lr)

        self.edge_idxs = [[0, 1, 2, 3], [3, 4, 5, 6],
                          [6, 7, 8, 9], [9, 10, 11, 0]]

        self.nonadjacent_patch_pairs = []
        self.adjacent_patch_pairs = []
        for i1, i2 in list(
                itertools.combinations(range(face_idxs.shape[0]), 2)):
            if args.wheels and (
                    i1 in [5, 11, 17, 23] or i2 in [5, 11, 17, 23]):
                continue
            if i2 in adjacencies[i1]:
                e1 = adjacencies[i1][i2]
                self.adjacent_patch_pairs.append((i1, i2, e1))
            else:
                self.nonadjacent_patch_pairs.append((i1, i2))

        self.d_points_to_tris = utils.PointToTriangleDistance.apply

        p_edge0, p_edge1, p_edge2, p_edge3 = [], [], [], []
        for i in range(self.n_samples_per_loop_side):
            for j in range(self.n_samples_per_loop_side):
                n = self.n_samples_per_loop_side

                if i > 0:
                    p_edge0.append(i + j*self.n_samples_per_loop_side)
                if i < n-1:
                    p_edge2.append(i + j*self.n_samples_per_loop_side)
                if j > 0:
                    p_edge3.append(i + j*self.n_samples_per_loop_side)
                if j < n-1:
                    p_edge1.append(i + j*self.n_samples_per_loop_side)
        self.grid_point_edges = (p_edge0, p_edge1, p_edge2, p_edge3)

        self.triangulation = []
        t_edge0, t_edge1, t_edge2, t_edge3 = [], [], [], []
        for i in range(self.n_samples_per_loop_side-1):
            for j in range(self.n_samples_per_loop_side-1):
                n = self.n_samples_per_loop_side-1

                if i > 1:
                    t_edge0.extend(
                        [len(self.triangulation), len(self.triangulation)+1])
                if i < n-2:
                    t_edge2.extend(
                        [len(self.triangulation), len(self.triangulation)+1])
                if j > 1:
                    t_edge3.extend(
                        [len(self.triangulation), len(self.triangulation)+1])
                if j < n-2:
                    t_edge1.extend(
                        [len(self.triangulation), len(self.triangulation)+1])

                self.triangulation.extend(
                    [[i + j*self.n_samples_per_loop_side,
                      i + (j+1)*self.n_samples_per_loop_side,
                      i + (j+1)*self.n_samples_per_loop_side + 1],
                     [i + j*self.n_samples_per_loop_side,
                      i + j*self.n_samples_per_loop_side + 1,
                      i + (j+1)*self.n_samples_per_loop_side + 1]])
        self.triangulation_edges = (t_edge0, t_edge1, t_edge2, t_edge3)

        _loss = Loss(args, self.triangulation_edges, self.triangulation,
                     self.grid_point_edges, self.n_samples_per_loop_side,
                     self.d_points_to_tris, self.edge_idxs,
                     self.nonadjacent_patch_pairs, self.adjacent_patch_pairs,
                     self.template_normals, symmetries)
        self._compute_losses = th.nn.DataParallel(
            _loss) if args.cuda else _loss

        if args.cuda:
            self.model.cuda()
            self._compute_losses.cuda()

    def forward(self, batch):
        ims = batch['ims']
        if self.args.cuda:
            ims = ims.cuda()

        params = self.model(ims)
        vertices, patches = utils.process_patches(
            params, self.vertex_idxs, self.face_idxs, self.edge_data,
            self.junctions, self.junction_order, self.vertex_t)

        st = th.empty(patches.shape[0], patches.shape[1],
                      self.n_samples_per_loop_side**2, 2).uniform_().to(params)

        points = utils.coons_sample(st[..., 0], st[..., 1], patches)
        normals = utils.coons_normals(st[..., 0], st[..., 1], patches)
        mtds = utils.coons_mtds(st[..., 0], st[..., 1], patches)

        return {'patches': patches, 'points': points, 'normals': normals,
                'mtds': mtds, 'st': st, 'params': params, 'vertices': vertices}

    def training_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        losses_dict = self._compute_losses(batch, self.forward(batch))
        loss = losses_dict['loss']
        loss.mean().backward()
        self.optimizer.step()

        return {k: v.mean().item() for k, v in losses_dict.items()}

    def init_validation(self):
        losses = ['loss', 'chamferloss', 'normalsloss', 'collisionloss',
                  'planarloss', 'templatenormalsloss', 'symmetryloss']
        ret = {loss: 0 for loss in losses}
        ret['count'] = 0
        return ret

    def validation_step(self, batch, running_data):
        self.model.eval()
        count = running_data['count']
        n = batch['ims'].shape[0]
        losses_dict = self._compute_losses(batch, self.forward(batch))
        loss = losses_dict['loss']
        chamferloss = losses_dict['chamferloss']
        normalsloss = losses_dict['normalsloss']
        collisionloss = losses_dict['collisionloss']
        planarloss = losses_dict['planarloss']
        templatenormalsloss = losses_dict['templatenormalsloss']
        symmetryloss = losses_dict['symmetryloss']
        return {
            'loss': (running_data['loss']*count +
                     loss.mean().item()*n) / (count+n),
            'chamferloss': (running_data['chamferloss']*count +
                            chamferloss.mean().item()*n) / (count+n),
            'normalsloss': (running_data['normalsloss']*count +
                            normalsloss.mean().item()*n) / (count+n),
            'collisionloss': (running_data['collisionloss']*count +
                              collisionloss.mean().item()*n) / (count+n),
            'planarloss': (running_data['planarloss']*count +
                           planarloss.mean().item()*n) / (count+n),
            'templatenormalsloss': (running_data['templatenormalsloss']*count +
                                    templatenormalsloss.mean().item()*n)
            / (count+n),
            'symmetryloss': (running_data['symmetryloss']*count
                             + symmetryloss.mean().item()*n) / (count+n),
            'count': count + n
        }


wheel_idxs = list(range(24))
no_wheel_idxs = list(range(24, 43))


class Loss(th.nn.Module):
    def __init__(self, args, triangulation_edges, triangulation,
                 grid_point_edges, n_samples_per_loop_side, d_points_to_tris,
                 edge_idxs, nonadjacent_patch_pairs, adjacent_patch_pairs,
                 template_normals, symmetries):
        super(Loss, self).__init__()
        self.args = args
        self.triangulation_edges = triangulation_edges
        self.triangulation = triangulation
        self.grid_point_edges = grid_point_edges
        self.d_points_to_tris = d_points_to_tris
        self.edge_idxs = edge_idxs
        self.nonadjacent_patch_pairs = nonadjacent_patch_pairs
        self.adjacent_patch_pairs = adjacent_patch_pairs
        self.symmetries = symmetries

        linspace = th.linspace(0, 1, n_samples_per_loop_side)
        s_grid, t_grid = th.meshgrid(linspace, linspace)
        self.s_grid = th.nn.Parameter(s_grid.flatten(), requires_grad=False)
        self.t_grid = th.nn.Parameter(t_grid.flatten(), requires_grad=False)
        self.template_normals = th.nn.Parameter(
            template_normals, requires_grad=False)

    def forward(self, batch, fwd_data):
        patches = fwd_data['patches']  # [b, n_patches, 12, 3]
        points = fwd_data['points']
        normals = fwd_data['normals']
        mtds = fwd_data['mtds']
        vertices = fwd_data['vertices']

        target_points = batch['points'].to(points)  # [b, n_points, 3]
        target_normals = batch['normals'].to(normals)
        if self.args.wheels:
            wheel_target_points = batch['wheel_points'].to(points)
            wheel_target_normals = batch['wheel_normals'].to(normals)

        b, n_patches, _, _ = patches.shape

        st = fwd_data['st']

        if self.symmetries is not None:
            xs, ys = self.symmetries
            xs_ = vertices[:, xs]
            ys_ = vertices[:, ys] * vertices.new_tensor([[[-1, 1, 1]]])
            symmetryloss = th.sum((xs_ - ys_)**2, dim=-1).mean()
        else:
            symmetryloss = patches.new_zeros(1)

        if self.args.wheels:
            mtds = mtds.view(b, -1)

            wheel_points = points[:, wheel_idxs]
            no_wheel_points = points[:, no_wheel_idxs]

            wheel_normals = normals[:, wheel_idxs]
            no_wheel_normals = normals[:, no_wheel_idxs]

            wheel_chamferloss_a, wheel_chamferloss_b, wheel_normalsloss_a, \
                wheel_normalsloss_b = utils.compute_chamfer_losses(
                    wheel_points, wheel_normals, wheel_target_points,
                    wheel_target_normals, self.args.w_normals > 0)
            no_wheel_chamferloss_a, no_wheel_chamferloss_b, \
                no_wheel_normalsloss_a, no_wheel_normalsloss_b = \
                utils.compute_chamfer_losses(no_wheel_points, no_wheel_normals,
                                             target_points, target_normals,
                                             self.args.w_normals > 0)

            chamferloss_a = th.cat(
                [wheel_chamferloss_a, no_wheel_chamferloss_a], dim=1)
            normalsloss_a = th.cat(
                [wheel_normalsloss_a, no_wheel_normalsloss_a], dim=1)

            ratio = batch['ratio']
            n_wheel_pts = wheel_target_points.shape[1]
            n_no_wheel_pts = target_points.shape[1]
            n_pts = n_wheel_pts + n_no_wheel_pts
            ratio_wheel = ratio * n_pts/n_wheel_pts
            ratio_no_wheel = (1-ratio) * n_pts/n_no_wheel_pts

            chamferloss_a = th.sum(mtds*chamferloss_a, dim=-1) / mtds.sum(-1)
            chamferloss_b = ratio_wheel * \
                wheel_chamferloss_b.mean(
                    1) + ratio_no_wheel * no_wheel_chamferloss_b.mean(1)
            chamferloss = ((chamferloss_a+chamferloss_b).mean() / 2).view(1)

            normalsloss_a = th.sum(mtds*normalsloss_a, dim=-1) / mtds.sum(-1)
            normalsloss_b = ratio_wheel * \
                wheel_normalsloss_b.mean(
                    1) + ratio_no_wheel * no_wheel_normalsloss_b.mean(1)
            normalsloss = ((normalsloss_a+normalsloss_b).mean() / 2).view(1)

        elif self.args.seperate_turbines:
            turbine_idxs = [x for x in range(76) if x not in [72, 73, 74, 75]]
            no_turbine_idxs = [x for x in range(76)
                               if x not in [10, 12, 14, 17, 19, 20, 21, 24, 25,
                                            27, 28, 29, 30, 31, 32, 33, 34,
                                            66]]

            turbine_points = points[batch['turbines']]
            turbine_points = turbine_points[:, turbine_idxs]
            no_turbine_points = points[batch['turbines']]
            no_turbine_points = no_turbine_points[:, no_turbine_idxs]

            turbine_normals = normals[batch['turbines']]
            turbine_normals = turbine_normals[:, turbine_idxs]
            no_turbine_normals = normals[batch['turbines']]
            no_turbine_normals = no_turbine_normals[:, no_turbine_idxs]

            turbine_mtds = mtds[batch['turbines']]
            turbine_mtds = turbine_mtds[:, turbine_idxs]
            no_turbine_mtds = mtds[batch['turbines']]
            no_turbine_mtds = no_turbine_mtds[:, no_turbine_idxs]

            turbine_patches = patches[batch['turbines']]
            turbine_patches = turbine_patches[:, turbine_idxs]
            no_turbine_patches = patches[batch['turbines']]
            no_turbine_patches = no_turbine_patches[:, no_turbine_idxs]

            turbine_target_points = target_points[batch['turbines']]
            no_turbine_target_points = target_points[batch['turbines']]

            turbine_target_normals = target_normals[batch['turbines']]
            no_turbine_target_normals = target_normals[batch['turbines']]

            if turbine_points.shape[0] > 0:
                turbine_mtds = turbine_mtds.view(turbine_mtds.shape[0], -1)
                turbine_chamferloss_a, turbine_chamferloss_b, \
                    turbine_normalsloss_a, turbine_normalsloss_b = \
                    utils.compute_chamfer_losses(turbine_points,
                                                 turbine_normals,
                                                 turbine_target_points,
                                                 turbine_target_normals,
                                                 self.args.w_normals > 0)
                turbine_chamferloss_a = th.sum(
                    turbine_mtds*turbine_chamferloss_a,
                    dim=-1) / turbine_mtds.sum(-1)
                turbine_chamferloss_b = turbine_chamferloss_b.mean(1)
                turbine_chamferloss = (
                    (turbine_chamferloss_a +
                     turbine_chamferloss_b).mean() / 2).view(1)
                turbine_normalsloss_a = th.sum(
                    turbine_mtds*turbine_normalsloss_a,
                    dim=-1) / turbine_mtds.sum(-1)
                turbine_normalsloss_b = turbine_normalsloss_b.mean(-1)
                turbine_normalsloss = (
                    (turbine_normalsloss_a +
                     turbine_normalsloss_b).mean() / 2).view(1)
            else:
                turbine_chamferloss = turbine_normalsloss = points.new_zeros(1)

            if no_turbine_points.shape[0] > 0:
                no_turbine_mtds = no_turbine_mtds.view(
                    no_turbine_mtds.shape[0], -1)
                no_turbine_chamferloss_a, no_turbine_chamferloss_b, \
                    no_turbine_normalsloss_a, no_turbine_normalsloss_b = \
                    utils.compute_chamfer_losses(no_turbine_points,
                                                 no_turbine_normals,
                                                 no_turbine_target_points,
                                                 no_turbine_target_normals,
                                                 self.args.w_normals > 0)
                no_turbine_chamferloss_a = th.sum(
                    no_turbine_mtds*no_turbine_chamferloss_a,
                    dim=-1) / no_turbine_mtds.sum(-1)
                no_turbine_chamferloss_b = no_turbine_chamferloss_b.mean(1)
                no_turbine_chamferloss = (
                    (no_turbine_chamferloss_a +
                     no_turbine_chamferloss_b).mean() / 2).view(1)
                no_turbine_normalsloss_a = th.sum(
                    no_turbine_mtds*no_turbine_normalsloss_a,
                    dim=-1) / no_turbine_mtds.sum(-1)
                no_turbine_normalsloss_b = no_turbine_normalsloss_b.mean(-1)
                no_turbine_normalsloss = (
                    (no_turbine_normalsloss_a +
                     no_turbine_normalsloss_b).mean() / 2).view(1)
            else:
                no_turbine_chamferloss = no_turbine_normalsloss = \
                    points.new_zeros(1)

            ratio_turbine = turbine_points.shape[0] / b
            chamferloss = ratio_turbine*turbine_chamferloss + \
                (1-ratio_turbine)*no_turbine_chamferloss
            normalsloss = ratio_turbine*turbine_normalsloss + \
                (1-ratio_turbine)*no_turbine_normalsloss

            del turbine_normals, turbine_target_normals, no_turbine_normals, \
                no_turbine_target_normals, turbine_points, \
                turbine_target_points, no_turbine_points, \
                no_turbine_target_points
        else:
            mtds = mtds.view(b, -1)

            chamferloss_a, chamferloss_b, normalsloss_a, normalsloss_b = \
                utils.compute_chamfer_losses(
                    points, normals, target_points, target_normals,
                    self.args.w_normals > 0)
            chamferloss_a = th.sum(mtds*chamferloss_a, dim=-1) / mtds.sum(-1)
            chamferloss_b = chamferloss_b.mean(1)
            chamferloss = ((chamferloss_a+chamferloss_b).mean() / 2).view(1)
            normalsloss_a = th.sum(mtds*normalsloss_a, dim=-1) / mtds.sum(-1)
            normalsloss_b = normalsloss_b.mean(-1)
            normalsloss = ((normalsloss_a+normalsloss_b).mean() / 2).view(1)

        mtds = mtds.view(b, n_patches, -1)

        if self.args.w_templatenormals:
            templatenormalsloss = th.sum(
                (self.template_normals - normals)**2, dim=-1)
            templatenormalsloss = th.sum(
                mtds*templatenormalsloss, dim=-1) / mtds.sum(-1)
        else:
            templatenormalsloss = th.zeros_like(chamferloss)

        del target_normals, normals, target_points

        if self.args.w_planar > 0:
            planarloss = utils.planar_patch_loss(st, points, mtds)
        else:
            planarloss = th.zeros_like(chamferloss)

        del points, mtds

        if self.args.w_collision > 0:
            collisionloss = chamferloss.new_zeros([b, 0])
            grid_points = utils.coons_sample(self.s_grid, self.t_grid, patches)
            triangles = grid_points[:, :, self.triangulation]

            i1s, i2s, e1s = zip(*self.adjacent_patch_pairs)
            points1 = grid_points[:, i1s]
            point_idxs = th.tensor([self.grid_point_edges[e]
                                    for e in e1s]).to(points1.device)
            point_idxs = point_idxs[None, :, :, None].expand(b, -1, -1, 3)
            points1 = th.gather(points1, 2, point_idxs)
            points2 = grid_points[:, i2s]

            triangles1 = triangles[:, i1s]
            triangle_idxs = th.tensor(
                [self.triangulation_edges[e] for e in e1s]
            ).to(triangles1.device)
            triangle_idxs = triangle_idxs[None, :, :,
                                          None, None].expand(b, -1, -1, 3, 3)
            triangles1 = th.gather(triangles1, 2, triangle_idxs)
            triangles2 = triangles[:, i2s]

            idxs = utils.bboxes_intersect(
                points1, points2, dim=2).any(0).nonzero().squeeze(1)
            n_adjacent_intersections = idxs.shape[0]

            if n_adjacent_intersections > 0:
                points1 = points1[:, idxs].view([-1] + list(points1.shape[2:]))
                points2 = points2[:, idxs].view([-1] + list(points2.shape[2:]))
                triangles1 = triangles1[:, idxs].view(
                    [-1] + list(triangles1.shape[2:]))
                triangles2 = triangles2[:, idxs].view(
                    [-1] + list(triangles2.shape[2:]))
                d1 = self.d_points_to_tris(points1, triangles2)
                d2 = self.d_points_to_tris(points2, triangles1)
                d = th.min(d1, d2).view(b, -1)
                collisionloss = th.cat(
                    [collisionloss, th.exp(-(d/self.args.sigma_collision)**2)],
                    dim=1)

            i1s, i2s = zip(*self.nonadjacent_patch_pairs)
            idxs = utils.bboxes_intersect(
                grid_points[:, i1s], grid_points[:, i2s], dim=2
            ).any(0).nonzero().squeeze(1)
            n_nonadjacent_intersections = idxs.shape[0]
            i1s = th.tensor(i1s).to(grid_points.device)[idxs]
            i2s = th.tensor(i2s).to(grid_points.device)[idxs]

            if n_nonadjacent_intersections > 0:
                points1 = grid_points[:, i1s].view(
                    [-1] + list(grid_points.shape[2:]))
                points2 = grid_points[:, i2s].view(
                    [-1] + list(grid_points.shape[2:]))
                triangles1 = triangles[:, i1s].view(
                    [-1] + list(triangles.shape[2:]))
                triangles2 = triangles[:, i2s].view(
                    [-1] + list(triangles.shape[2:]))
                d1 = self.d_points_to_tris(points1, triangles2)
                d2 = self.d_points_to_tris(points2, triangles1)
                d = th.min(d1, d2).view(b, -1)
                collisionloss = th.cat(
                    [collisionloss, th.exp(-(d/self.args.sigma_collision)**2)],
                    dim=1)

            del triangles

            if n_adjacent_intersections + n_nonadjacent_intersections > 0:
                collisionloss = collisionloss.sum(-1).mean()
            else:
                collisionloss = th.zeros_like(chamferloss)
        else:
            collisionloss = th.zeros_like(chamferloss)

        loss = chamferloss + self.args.w_normals*normalsloss + \
            self.args.w_collision*collisionloss + \
            self.args.w_planar*planarloss + \
            self.args.w_templatenormals*templatenormalsloss + \
            self.args.w_symmetry*symmetryloss

        return {
            'loss': loss,
            'chamferloss': chamferloss,
            'normalsloss': normalsloss,
            'collisionloss': collisionloss,
            'planarloss': planarloss,
            'templatenormalsloss': templatenormalsloss,
            'symmetryloss': symmetryloss,
        }

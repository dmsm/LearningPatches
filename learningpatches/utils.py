import numpy as np
import torch as th

from .coons_utils import *


def subbezier(t1, t2, params):
    """Compute control points for cubic Bezier curve between t1 and t2.

    t1 -- [batch_size]
    t2 -- [batch_size]
    params -- [batch_size, 4, 3]
    """
    def dB_dt(t):
        return params[:, 0]*(-3*(1-t)**2) + params[:, 1]*(3*(1-4*t+3*t**2)) \
            + params[:, 2]*(3*(2*t-3*t**2)) + params[:, 3]*(3*t**2)

    t1 = t1[:, None]
    t2 = t2[:, None]
    sub_pts = th.empty_like(params)
    sub_pts[:, 0] = bezier_sample(t1[:, :, None], params).squeeze(1)
    sub_pts[:, 3] = bezier_sample(t2[:, :, None], params).squeeze(1)
    sub_pts[:, 1] = (t2-t1)*dB_dt(t1)/3 + sub_pts[:, 0]
    sub_pts[:, 2] = sub_pts[:, 3] - (t2-t1)*dB_dt(t2)/3
    return sub_pts


def bezier_sample(t, params):
    """Sample points from cubic Bezier curves defined by params at t values."""
    A = params.new_tensor([[1, 0, 0, 0],
                           [-3, 3, 0, 0],
                           [3, -6, 3, 0],
                           [-1, 3, -3, 1]])

    t = t.pow(t.new_tensor([0, 1, 2, 3]))  # [n_samples, 4]

    points = t @ A @ params  # [..., n_samples, 3]
    return points


def coons_sample(s, t, params):
    """Sample points from Coons patch defined by params at s, t values.

    params -- [..., 12, 3]
    """
    sides = [params[..., :4, :], params[..., 3:7, :],
             params[..., 6:10, :], params[..., [9, 10, 11, 0], :]]
    corners = [params[..., [0], :], params[..., [3], :],
               params[..., [9], :], params[..., [6], :]]

    s = s[..., None]
    t = t[..., None]
    B = corners[0] * (1-s) * (1-t) + corners[1] * s * (1-t) + \
        corners[2] * (1-s) * t + corners[3] * s * t  # [..., n_samples, 3]

    Lc = bezier_sample(s, sides[0]) * (1-t) + bezier_sample(1-s, sides[2]) * t
    Ld = bezier_sample(t, sides[1]) * s + bezier_sample(1-t, sides[3]) * (1-s)
    return Lc + Ld - B


def batched_cdist_l2(x1, x2):
    """Compute batched l2 cdist."""
    x1_norm = x1.pow(2).sum(-1, keepdim=True)
    x2_norm = x2.pow(2).sum(-1, keepdim=True)
    res = th.baddbmm(
        x2_norm.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm).clamp_min_(1e-10).sqrt_()
    return res


def planar_patch_loss(params, points, mtds):
    """Compute planar patch loss from control points, samples, and Jacobians.

    params -- [..., 2]
    points -- [..., 3]
    """
    X = th.cat([params.new_ones(list(params.shape[:-1]) + [1]), params],
               dim=-1)
    b = th.inverse(X.transpose(-1, -2) @ X) @ X.transpose(-1, -2) @ points
    distances = (X @ b - points).pow(2).sum(-1)
    return th.sum(distances*mtds, dim=-1) / mtds.sum(-1)


def dot(a, b):
    """Dot product."""
    return th.sum(a*b, dim=-1, keepdim=True)


def dot2(a):
    """Squared norm."""
    return dot(a, a)


def d_points_to_tris(points, triangles):
    """Compute distance frome each point to the corresponding triangle.

    points -- [b, n, 3]
    triangles -- [b, n, 3, 3]
    """
    v21 = triangles[:, :, 1]-triangles[:, :, 0]
    v32 = triangles[:, :, 2]-triangles[:, :, 1]
    v13 = triangles[:, :, 0]-triangles[:, :, 2]
    p1 = points - triangles[:, :, 0]
    p2 = points - triangles[:, :, 1]
    p3 = points - triangles[:, :, 2]
    nor = th.cross(v21, v13, dim=-1)

    cond = dot(th.cross(v21, nor, dim=-1), p1).sign() \
        + dot(th.cross(v32, nor, dim=-1), p2).sign() \
        + dot(th.cross(v13, nor, dim=-1), p3).sign() < 2
    cond = cond.float()
    result = cond * th.stack([
        dot2(v21 * th.clamp(dot(v21, p1) / dot2(v21), 0, 1) - p1),
        dot2(v32 * th.clamp(dot(v32, p2) / dot2(v32), 0, 1) - p2),
        dot2(v13 * th.clamp(dot(v13, p3) / dot2(v13), 0, 1) - p3)
    ], dim=-1).min(-1)[0] + (1-cond) * dot(nor, p1) * dot(nor, p1) / dot2(nor)
    return result.squeeze(-1).min(-1)[0]


class PointToTriangleDistance(th.autograd.Function):
    """Autograd function for computing smallest point to triangle distance."""

    @staticmethod
    def forward(ctx, points, triangles):
        """Compute smallest distance between each point and triangle batch.

        points -- [batch_size, n_points, 3]
        triangles -- [batch_size, n_triagles, 3, 3]
        """
        b = points.shape[0]

        v21 = triangles[:, None, :, 1]-triangles[:, None, :, 0]
        v32 = triangles[:, None, :, 2]-triangles[:, None, :, 1]
        v13 = triangles[:, None, :, 0]-triangles[:, None, :, 2]
        p1 = points[:, :, None] - triangles[:, None, :, 0]
        p2 = points[:, :, None] - triangles[:, None, :, 1]
        p3 = points[:, :, None] - triangles[:, None, :, 2]
        nor = th.cross(v21, v13, dim=-1)

        cond = dot(th.cross(v21, nor, dim=-1), p1).sign() \
            + dot(th.cross(v32, nor, dim=-1), p2).sign() \
            + dot(th.cross(v13, nor, dim=-1), p3).sign() < 2
        cond = cond.float()
        result = cond * th.stack([
            dot2(v21 * th.clamp(dot(v21, p1) / dot2(v21), 0, 1) - p1),
            dot2(v32 * th.clamp(dot(v32, p2) / dot2(v32), 0, 1) - p2),
            dot2(v13 * th.clamp(dot(v13, p3) / dot2(v13), 0, 1) - p3)
        ], dim=-1).min(-1)[0] + (1-cond) \
            * dot(nor, p1) * dot(nor, p1) / dot2(nor)
        result = result.squeeze(-1)

        _, nearest_tris_idxs = result.min(-1)  # [b, n_points]
        _, nearest_points_idxs = result.min(-2)  # [b, n_tris]
        ctx.save_for_backward(
            points, triangles, nearest_tris_idxs, nearest_points_idxs)

        return result.view(b, -1).min(-1)[0]

    @staticmethod
    def backward(ctx, grad_output):
        """Only consider the closest point-triangle pair for gradient."""
        points, triangles, nearest_tris_idxs, nearest_points_idxs = \
            ctx.saved_tensors
        grad_points = grad_tris = None

        if ctx.needs_input_grad[0]:
            idx = nearest_tris_idxs[..., None, None].expand(
                list(nearest_tris_idxs.shape) + [3, 3])
            nearest_tris = triangles.gather(index=idx, dim=1)
            with th.enable_grad():
                distance = d_points_to_tris(points, nearest_tris)
                grad_points = th.autograd.grad(outputs=distance, inputs=points,
                                               grad_outputs=grad_output,
                                               only_inputs=True)[0]
        if ctx.needs_input_grad[1]:
            idx = nearest_points_idxs[..., None].expand(
                list(nearest_points_idxs.shape) + [3])
            nearest_points = points.gather(index=idx, dim=1)
            with th.enable_grad():
                distance = d_points_to_tris(nearest_points, triangles)
                grad_tris = th.autograd.grad(outputs=distance,
                                             inputs=triangles,
                                             grad_outputs=grad_output,
                                             only_inputs=True)[0]

        return grad_points, grad_tris


def bboxes_intersect(points1, points2, dim=1):
    """Compute whether bounding boxes of two point clouds intersect."""
    min1 = points1.min(dim)[0]
    max1 = points1.max(dim)[0]
    min2 = points2.min(dim)[0]
    max2 = points2.max(dim)[0]
    center1 = (min1 + max1)/2
    center2 = (min2 + max2)/2
    size1 = max1 - min1
    size2 = max2 - min2
    return ((center1 - center2).abs() * 2 <= size1 + size2).all(-1)


def logit(x):
    """Inverse of softmax."""
    return np.log(x / (1-x))


def compute_chamfer_losses(points, normals, target_points, target_normals,
                           compute_normals=True):
    """Compute area-weighted Chamfer and normals losses."""
    b = points.shape[0]
    points = points.view(b, -1, 3)

    # [b, n_total_samples, n_points]
    distances = batched_cdist_l2(points, target_points)

    chamferloss_a, idx_a = distances.min(2)  # [b, n_total_samples]
    chamferloss_b, idx_b = distances.min(1)

    if compute_normals:
        normals = normals.view(b, -1, 3)

        # [b, n_total_samples, 1, 3]
        idx_a = idx_a[..., None, None].expand(-1, -1, -1, 3)
        nearest_target_normals = \
            target_normals[:, None].expand(list(distances.shape) + [3]) \
            .gather(index=idx_a, dim=2).squeeze(2)  # [b, n_total_samples, 3]

        # [b, 1, n_points, 3]
        idx_b = idx_b[..., None, :, None].expand(-1, -1, -1, 3)
        nearest_normals = \
            normals[:, :, None].expand(list(distances.shape) + [3]) \
            .gather(index=idx_b, dim=1).squeeze(1)  # [b, n_points, 3]

        normalsloss_a = th.sum((nearest_target_normals - normals)**2, dim=-1)
        normalsloss_b = th.sum((nearest_normals - target_normals)**2, dim=-1)
    else:
        normalsloss_a = th.zeros_like(chamferloss_a)
        normalsloss_b = th.zeros_like(chamferloss_b)

    return chamferloss_a, chamferloss_b, normalsloss_a, normalsloss_b


def process_patches(params, vertex_idxs, face_idxs, edge_data, junctions,
                    junction_order, vertex_t):
    """Process all junction curves to compute explicit patch control poitns."""
    vertices = params.clone()[:, vertex_idxs]

    for i in junction_order:
        edge = junctions[i]
        t = th.sigmoid(params[:, vertex_t[i]])
        vertex = bezier_sample(t[:, None, None], vertices[:, edge]).squeeze(1)
        vertices = vertices.clone()
        vertices[:, i] = vertex

        for a, b, c, d in edge_data[i]:
            if a not in junctions:
                a, b, c, d = d, c, b, a

            edge = junctions[a]
            t_a = th.sigmoid(params[:, vertex_t[a]])
            v0_a, _, _, v3_a = edge
            if d == v0_a:
                t_d = th.zeros_like(t_a)
            elif d == v3_a:
                t_d = th.ones_like(t_a)
            else:
                v0_d, _, _, v3_d = junctions[d]
                t_d = th.sigmoid(params[:, vertex_t[d]])
                if v0_a == v0_d and v3_a == v3_d:
                    pass
                elif v0_a == v3_d and v3_a == v0_d:
                    t_d = 1 - t_d
                else:
                    edge = junctions[d]
                    if a == v0_d:
                        t_a = th.zeros_like(t_d)
                    elif a == v3_d:
                        t_a = th.ones_like(t_d)

            curve = subbezier(t_a, t_d, vertices[:, edge])[:, 1:-1]
            vertices = vertices.clone()
            vertices[:, [b, c]] = curve

    patches = vertices[:, face_idxs]

    return vertices, patches


def tri_area(tri):
    """Compute the area of a triangle form its vertices."""
    a = tri[:, 0]
    b = tri[:, 1]
    c = tri[:, 2]
    ab = b - a
    ac = c - a
    return th.cross(ab, ac).abs().norm(2, dim=1) / 2


def write_obj(file, patches, res=30):
    """Write Coons patches to an obj file."""
    linspace = th.linspace(0, 1, res).to(patches)
    s_grid, t_grid = th.meshgrid(linspace, linspace)
    verts = coons_sample(s_grid.flatten(),
                         t_grid.flatten(), patches).cpu().numpy()
    n_verts = verts.shape[1]
    with open(file, 'w') as f:
        for p, patch in enumerate(verts):
            for x, y, z in patch:
                f.write(f'v {x} {y} {z}\n')
            for i in range(res-1):
                for j in range(res-1):
                    f.write(
                        f'f {i*res + j+2 + p*n_verts} {i*res + j+1 + p*n_verts} {(i+1)*res + j+1 + p*n_verts}\n')
                    f.write(
                        f'f {(i+1)*res + j+2 + p*n_verts} {i*res + j+2 + p*n_verts} {(i+1)*res + j+1 + p*n_verts}\n')

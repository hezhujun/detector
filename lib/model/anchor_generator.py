import numpy as np


def generate_anchor(stride, scales, ratios, size):
    """

    :param stride: 步长，每个bin的长度
    :param scales: 面积 tuple
    :param ratios: 宽高比 tuple
    :param size: 网格的大小 (w, h)
    :return:
    """
    scales = np.array(scales, dtype=np.float32).reshape((-1, 1))
    ratios = np.array(ratios, dtype=np.float32)
    hs = np.sqrt(scales / ratios)  # shape (n_s, n_r)
    ws = hs * ratios               # shape (n_s, n_r)
    hs = hs.reshape((-1))
    ws = ws.reshape((-1))
    w_h = np.stack([ws, hs], axis=1)  # shape (n_s * n_r, 2)

    center_point = np.array((stride // 2, stride // 2), dtype=np.float32).reshape((1, 2))
    x1_y1 = center_point - w_h / 2
    x2_y2 = center_point + w_h / 2

    anchor_offset = np.concatenate([x1_y1, x2_y2], axis=1)  # shape (n_s * n_r, 4)

    grid_x = np.arange(size[0], dtype=np.float32)
    grid_y = np.arange(size[1], dtype=np.float32)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)

    grid = np.stack([grid_x, grid_y, grid_x, grid_y], axis=2)  # shape (w, h, 4)
    grid = grid * stride

    # grid (w, h, 4) -> (w, h, 1,         4)
    # anchor_offset     (      n_s * n_r, 4)
    w, h, _ = grid.shape
    grid = grid.reshape((w, h, 1, 4))
    anchors = grid + anchor_offset  # shape (w, h, n_s * n_r, 4)

    return anchors


if __name__ == '__main__':
    generate_anchor(2**4, (32**2, 64**2, 128**2), (0.5, 1, 2), (4, 5))
    # generate_anchor(2**4, (32**2,), (1,), (4, 5))

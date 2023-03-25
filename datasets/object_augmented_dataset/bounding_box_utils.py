import torch


def masks_to_bboxes(mask, fmt='c'):

    """ Convert a mask tensor to one or more bounding boxes.
    Note: This function is a bit new, make sure it does what it says.  /Andreas
    :param mask: Tensor of masks, shape = (..., H, W)
    :param fmt: bbox layout. 'c' => "center + size" or (x_center, y_center, width, height)
                             't' => "top left + size" or (x_left, y_top, width, height)
                             'v' => "vertices" or (x_left, y_top, x_right, y_bottom)
    :return: tensor containing a batch of bounding boxes, shape = (..., 4)
    """
    batch_shape = mask.shape[:-2]
    mask = mask.reshape((-1, *mask.shape[-2:]))
    bboxes = []

    for m in mask:
        mx = torch.nonzero(m.sum(dim=-2))
        my = torch.nonzero(m.sum(dim=-1))
        bb = [mx.min(), my.min(), mx.max(), my.max()] if (len(mx) > 0 and len(my) > 0) else [0, 0, 0, 0]
        bboxes.append(bb)

    bboxes = torch.tensor(bboxes, dtype=torch.float32, device=mask.device)
    bboxes = bboxes.reshape(batch_shape + (4,))

    if fmt == 'v':
        return bboxes

    x1 = bboxes[..., :2]
    s = bboxes[..., 2:] - x1 + 1

    if fmt == 'c':
        return torch.cat((x1 + 0.5 * s, s), dim=-1)
    elif fmt == 't':
        return torch.cat((x1, s), dim=-1)

    raise ValueError("Undefined bounding box layout '%s'" % fmt)


def masks_to_bboxes_multi(mask, ids, fmt='c'):
    assert mask.dim() == 2
    bboxes = []

    for id in ids:
        mx = (mask == id).sum(dim=-2).nonzero()
        my = (mask == id).float().sum(dim=-1).nonzero()
        bb = [mx.min(), my.min(), mx.max(), my.max()] if (len(mx) > 0 and len(my) > 0) else [0, 0, 0, 0]

        bb = torch.tensor(bb, dtype=torch.float32, device=mask.device)

        x1 = bb[:2]
        s = bb[2:] - x1 + 1

        if fmt == 'v':
            pass
        elif fmt == 'c':
            bb = torch.cat((x1 + 0.5 * s, s), dim=-1)
        elif fmt == 't':
            bb = torch.cat((x1, s), dim=-1)
        else:
            raise ValueError("Undefined bounding box layout '%s'" % fmt)
        bboxes.append(bb)

    return bboxes

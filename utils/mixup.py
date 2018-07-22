import torch
import numpy


def shuffle_minibatch(inputs, targets, args,mixup=True):
    """Shuffle a minibatch and do linear interpolation between images and labels.
    Args:
        inputs: a numpy array of images with size batch_size x H x W x 3.
        targets: a numpy array of labels with size batch_size x 1.
        mixup: a boolen as whether to do mixup or not. If mixup is True, we
            sample the weight from beta distribution using parameter alpha=1,
            beta=1. If mixup is False, we set the weight to be 1 and 0
            respectively for the randomly shuffled mini-batches.
    """
    # targets.unsqueeze_(1)
    # print(inputs.shape)
    # print(targets.shape)

    batch_size = inputs.shape[0]
    rp1 = torch.randperm(batch_size)
    inputs1 = inputs[rp1]
    targets1 = targets[rp1]
    targets1_1 = targets1.unsqueeze(1)

    rp2 = torch.randperm(batch_size)
    inputs2 = inputs[rp2]
    targets2 = targets[rp2]
    targets2_1 = targets2.unsqueeze(1)

    y_onehot = torch.FloatTensor(batch_size, 20)
    y_onehot.zero_()
    targets1_oh = y_onehot.scatter_(1, targets1_1, 1)

    y_onehot2 = torch.FloatTensor(batch_size, 20)
    y_onehot2.zero_()
    targets2_oh = y_onehot2.scatter_(1, targets2_1, 1)

    if mixup is True:
        a = numpy.random.beta(1, 1, [batch_size, 1])
    else:
        a = numpy.ones((batch_size, 1))

    b = numpy.tile(a[..., None, None], [1, 3, args.size, args.size])

    inputs1 = inputs1 * torch.from_numpy(b).float()
    inputs2 = inputs2 * torch.from_numpy(1 - b).float()

    c = numpy.tile(a, [1, 20])
    targets1_oh = targets1_oh.float() * torch.from_numpy(c).float()
    targets2_oh = targets2_oh.float() * torch.from_numpy(1 - c).float()

    inputs_shuffle = inputs1 + inputs2
    targets_shuffle = targets1_oh + targets2_oh

    return inputs_shuffle, targets_shuffle

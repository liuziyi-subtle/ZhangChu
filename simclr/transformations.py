import torchvision


class TransformsSimCLR:
    """
    对信号做随机变换，形成positive pair->x_i, x_j.
    """
    # TODO: 设置随机变换的参数.

    def __init__(self):
        self.train_transform = torchvision.transforms.Compose(
            [
                AddNoise(80),
                TimeWarping(20)
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)


class AddNoise(object):
    """
    add random noise on original signal x.
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x):
        return x + self.add_noise(x, self.alpha)

    def add_noise(x, alpha):
        """ add gaussian noise on original signal.
        """
        if not isinstance(x, (np.ndarray, pd.Series)):
            x = np.asarray(x)
        E_avg_pow = np.dot(x, x)  # average power of S(t)
        N = [random.gauss(0, np.sqrt(10**((E_avg_pow - alpha) / 10)))
             for i in range(len(x))]
        return N


class TimeWarping(object):
    """
    randomly sample and scratch/squeeze segments in signal.
    """

    def __init__(self, num_segments):
        self.num_segments = num_segments

    def __call__(self, x):
        return self.time_warping(x, self.num_segments)

    def time_warping(x, num_segments):
        segment_size = int(x / num_segments)
        segment_order = permutation(num_segments)
        num_samples = ([int(segment_size / 2)
                        if so < num_segments / 2 else int(segment_size * 2)])

        arr_list = []
        for index, ns in enumerate(num_samples):
            segment_range = range(index * segment_size,
                                  (index + 1) * segment_size)
            segment_trans = signal.resample(x[segment_range], ns, axis=1)
            arr_list.append(segment_trans)
        x_time_warping = np.concatenate(arr_list)
        return x_time_warping[:len(x)]

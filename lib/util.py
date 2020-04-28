import torch.optim.lr_scheduler as lr_scheduler


class WarmingUpScheduler(lr_scheduler.LambdaLR):

    def __init__(self, optimizer, init_factor=0.1, steps=100):
        assert init_factor >= 0 and init_factor <= 1
        assert steps > 1
        def f(epochs):
            if epochs < steps:
                return (1 - init_factor) / (steps - 1) * epochs + init_factor
            else:
                return 1
        super(WarmingUpScheduler, self).__init__(optimizer, f)

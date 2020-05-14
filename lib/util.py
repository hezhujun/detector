import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from PIL import ImageDraw


class WarmingUpScheduler(lr_scheduler.LambdaLR):

    def __init__(self, optimizer, init_factor=0.1, steps=100):
        assert init_factor >= 0 and init_factor <= 1
        assert steps > 1
        self.steps = steps
        def f(epochs):
            if epochs < steps:
                return (1 - init_factor) / (steps - 1) * epochs + init_factor
            else:
                return 1
        super(WarmingUpScheduler, self).__init__(optimizer, f)
        
    def step(self, epoch=None):
        if epoch and epoch < self.steps or self.last_epoch + 1 < self.steps:
            super(WarmingUpScheduler, self).step(epoch)


class SummaryWriterWrap(SummaryWriter):
    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10,
                 flush_secs=120, filename_suffix=''):
        super(SummaryWriterWrap, self).__init__(log_dir=log_dir,
                                                comment=comment,
                                                purge_step=purge_step,
                                                max_queue=max_queue,
                                                flush_secs=flush_secs,
                                                filename_suffix=filename_suffix)
        self.cur_epoch = 0

    def step(self, epoch=None):
        if epoch is None:
            self.cur_epoch += 1
        else:
            self.cur_epoch = epoch

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        step = global_step if global_step else self.cur_epoch
        super(SummaryWriterWrap, self).add_scalar(tag, scalar_value, global_step=step, walltime=walltime)


def draw_image(image, boxes, labels=None, scores=None):
    """

    :param image: PIL.Image.Image
    :param boxes: numpy ndarray (N, 4): (x1, y1, x2, y2)
    :param labels: numpy ndarray (N,)
    :param scores: numpy ndarray (N,)
    :return:
    """
    draw = ImageDraw.Draw(image, mode="RGBA")
    N = len(boxes)
    for i in range(N):
        x1, y1, x2, y2 = boxes[i]
        #x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        xy = [(x1, y1), (x2, y2)]
        draw.rectangle(xy, fill=(0, 0, 0, 0), outline=(0xff, 0, 0, 0xff))

    return image

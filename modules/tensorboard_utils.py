import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tensorboardX import SummaryWriter
from tensorboardX.summary import hparams


class CustomSummaryWriter(SummaryWriter):
    def __init__(self, logdir=None, comment='', purge_step=None,
                 max_queue=10, flush_secs=120, filename_suffix='',
                 write_to_disk=True, log_dir=None, **kwargs):
        super().__init__(logdir, comment, purge_step, max_queue,
                         flush_secs, filename_suffix, write_to_disk,
                         log_dir, **kwargs)

    def add_hparams(self, hparam_dict=None, metric_dict=None, name=None, global_step=None):
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v, global_step)


class TensorBoardUtils(object):
    def __init__(self, tensorboard_writer: SummaryWriter):
        super(TensorBoardUtils, self).__init__()
        self.tensorboard_writer = tensorboard_writer

    def addPlotConfusionMatrix(self, dataXY, ticks, tag, global_step=0, is_percent_added=False):
        fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        res = ax.imshow(dataXY, cmap=plt.cm.Blues if is_percent_added else plt.cm.binary,
                        interpolation='nearest')

        width, height = dataXY.shape

        if is_percent_added:
            cmap_min, cmap_max = res.cmap(0), res.cmap(256)
            thresh = (dataXY.max() + dataXY.min()) / 2.0
            for x in range(width):
                for y in range(height):
                    color = cmap_max if dataXY[x][y] < thresh else cmap_min
                    ax.annotate(str(round(float(dataXY[x][y]), 3)), xy=(y, x),  # percent without decimals
                                horizontalalignment='center',
                                verticalalignment='center',
                                color=color)

        cb = fig.colorbar(res)
        if ticks is not None:
            plt.xticks(range(width), ticks, rotation=90)
            plt.yticks(range(height), ticks)

        plt.xlabel('Predicted class')
        plt.ylabel('Actual class')

        fig.set_tight_layout(True)
        canvas = FigureCanvas(fig)
        canvas.draw()

        # width, height = fig.get_size_inches() * fig.get_dpi()
        width, height = fig.canvas.get_width_height()
        image = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)

        image = np.swapaxes(image, 2, 0)
        image = np.swapaxes(image, 2, 1)
        self.tensorboard_writer.add_image(tag=tag, img_tensor=image, global_step=global_step)

        plt.close(fig)

    def addHistogramsTwo(self, data_positives, data_negatives, tag, global_step=0):
        fig = plt.figure()
        plt.clf()

        n, bins, patches = plt.hist(data_positives, 100, density=True, facecolor='g', alpha=0.75)
        n, bins, patches = plt.hist(data_negatives, 100, density=True, facecolor='r', alpha=0.75)

        plt.xlabel('Distance')
        plt.ylabel('Samples')

        fig.set_tight_layout(True)
        canvas = FigureCanvas(fig)
        canvas.draw()

        # width, height = fig.get_size_inches() * fig.get_dpi()
        width, height = fig.canvas.get_width_height()
        image = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)

        image = np.swapaxes(image, 2, 0)
        image = np.swapaxes(image, 2, 1)
        self.tensorboard_writer.add_image(tag=tag, img_tensor=image, global_step=global_step)

        plt.close(fig)

    def addPlot2D(self, dataXY, tag, global_step=0):
        dataXY = np.copy(dataXY)
        min_value = np.min(dataXY)
        dataXY += abs(min_value)

        max_value = np.max(dataXY)
        dataXY = dataXY / max_value

        dataXY *= 255
        dataXY = dataXY.astype(dtype=np.uint8)

        image = np.transpose(dataXY) # H, W
        image = np.expand_dims(image, axis=2)
        image = np.tile(image, (1,1,3))

        image = np.swapaxes(image, 2, 0)
        image = np.swapaxes(image, 2, 1)
        self.tensorboard_writer.add_image(tag=tag, img_tensor=image, global_step=global_step)

    def addImageWith_mask(self, image, mask, tag, global_step=0):
        image = np.copy(image)
        min_value = np.min(image)
        image += abs(min_value)

        max_value = np.max(image)
        image = image / max_value

        image *= 255
        image = image.astype(dtype=np.uint8)

        if len(image.shape) == 3:
            image = np.transpose(image, (1, 2, 0))
        else:
            image = np.expand_dims(image, axis=2)
            image = np.tile(image, (1,1,3))
        # add mask
        masked_layer = image.copy()
        masked_layer[np.array(mask, dtype=bool)] = np.array((0,0,100), dtype=np.uint8)
        image = np.array(image * 0.5 + masked_layer * 0.5, dtype=np.uint8)

        image = np.swapaxes(image, 2, 0)
        image = np.swapaxes(image, 2, 1)

        self.tensorboard_writer.add_image(tag=tag, img_tensor=image, global_step=global_step)

    def addPlot1D(self, data, tag, global_step=0, axis_labels=None):
        data = np.copy(data)
        fig = plt.figure()

        if not axis_labels is None:
            plt.xlabel(axis_labels[0])
            plt.ylabel(axis_labels[1])
            if len(data) > 1:
                plt.plot(data[0], data[1])
        else:
            plt.plot(data)

        fig.set_tight_layout(True)
        canvas = FigureCanvas(fig)
        canvas.draw()

        # width, height = fig.get_size_inches() * fig.get_dpi()
        width, height = fig.canvas.get_width_height()
        image = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)

        image = np.swapaxes(image, 2, 0)
        image = np.swapaxes(image, 2, 1)
        self.tensorboard_writer.add_image(tag=tag, img_tensor=image, global_step=global_step)

        plt.close(fig)

    def addAudio(self, data, tag, global_step=0, sample_rate=16000):
        data = np.copy(data)
        self.tensorboard_writer.add_audio(tag=tag, snd_tensor=data, global_step=global_step, sample_rate=sample_rate)
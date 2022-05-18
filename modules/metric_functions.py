import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os


def dice_coeficient(predict, target):
    smooth = 1e-5
    classes = predict.shape[1]  # [B, C, W, H]
    coef_stack = []
    for class_idx in range(classes):
        predict_item = predict[:, class_idx, :, :].clone()
        target_item = target[:, class_idx, :, :].clone()

        target_sum_list = torch.sum(target_item, dim=[1, 2]).cpu().numpy()
        idxs_zeros = (target_sum_list == 0).nonzero()[0]
        if len(idxs_zeros) > 0:
            predict_item[idxs_zeros] = 1 - predict_item[idxs_zeros]
            target_item[idxs_zeros] = torch.ones_like(target_item[idxs_zeros])

        intersection = torch.sum(predict_item * target_item)
        target_sum = torch.sum(target_item)
        predict_sum = torch.sum(predict_item)
        coef = 2. * (intersection + smooth) / (predict_sum + target_sum + smooth)
        coef_stack.append(coef)
    coef_mean = torch.mean(torch.stack(coef_stack))
    return coef_mean


def IoU(predict, target):
    smooth = 1e-5
    classes = predict.shape[1]  # [B, C, W, H]
    coef_stack = []
    for class_idx in range(classes):
        predict_item = predict[:, class_idx, :, :].clone()
        target_item = target[:, class_idx, :, :].clone()

        target_sum_list = torch.sum(target_item, dim=[1, 2]).cpu().numpy()
        idxs_zeros = (target_sum_list == 0).nonzero()[0]
        if len(idxs_zeros) > 0:
            predict_item[idxs_zeros] = 1 - predict_item[idxs_zeros]
            target_item[idxs_zeros] = torch.ones_like(target_item[idxs_zeros])

        intersection = torch.sum(predict_item * target_item)
        target_sum = torch.sum(target_item)
        predict_sum = torch.sum(predict_item)
        coef = (intersection + smooth) / (predict_sum + target_sum - intersection + smooth)
        # print(f'{class_idx} coef: {coef}')
        coef_stack.append(coef)
    # print(f'coef_stack: {coef_stack}')
    # print(torch.stack(coef_stack))
    coef_classes = torch.stack(coef_stack)
    coef_mean = torch.mean(coef_classes)
    return coef_mean, coef_classes


def make_figures(x, y, y_prim, class_names, epoch, metrics, args):
    plt.clf()
    # plt.figure(1, figsize=(12,5))
    # plt.rcParams["figure.figsize"] = (20, 5)
    # plt.subplot(121)  # row col idx
    ax = plt.subplot(131)  # row col idx
    plts = []
    c = 0
    # plt.figure('Metrics')
    for key, value in metrics.items():
        plts += plt.plot(value, f'C{c}', label=key)
        # plt.xlim(0, args.epochs)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax = plt.twinx()

        c += 1
    plt.tick_params(
        axis='y',
        which='both',
        right=False,
        labelright=False
    )
    plt.legend(plts, [it.get_label() for it in plts])

    # print(x.shape, np_y.shape)
    # plt.figure('Images')
    # plt.clf()
    num_figs = 4 if x.shape[0] >= 4 else x.shape[0]
    classes = len(class_names)
    for i in range(num_figs):
        rows = 1 + classes
        cols = 3 * num_figs
        im_idx_i = num_figs + 2 * i + 1
        plt.subplot(rows, cols, im_idx_i)
        # print(rows, cols)
        plt.title(f'input', pad=1)
        plt.imshow(x[i, 0, :, :], cmap=plt.get_cmap('gray'))
        plt.axis('off')
        #
        for j in range(classes):
            im_idx_j = cols * (j + 1) + im_idx_i
            # print(im_idx)
            axt = plt.subplot(rows, cols, im_idx_j)
            if j == 0:
                plt.title('target', pad=1)
            # plt.title(f'{class_names[j]} y', fontsize='small', pad=1)
            plt.imshow(y[i, j, :, :])
            # plt.tick_params(
            #     axis='both',
            #     which='both',
            #     botton=False, top=False,
            #     left=False, right=False,
            #     labelbottom=False, labelsleft=False
            # )
            plt.xticks([]), plt.yticks([])
            axt.spines['top'].set_visible(False)
            axt.spines['right'].set_visible(False)
            axt.spines['bottom'].set_visible(False)
            axt.spines['left'].set_visible(False)
            if i == 0:
                plt.ylabel(f'{class_names[j]}', fontsize='large')

            plt.subplot(rows, cols, im_idx_j + 1)
            if j == 0:
                plt.title('predict', pad=1)
            # plt.title(f"{class_names[j]} y'", fontsize='small', pad=1)
            plt.imshow(y_prim[i, j, :, :])
            plt.axis('off')
    print(plt.gcf().get_size_inches())
    plt.tight_layout(pad=0.01)
    path_run_fig = f'./results/{args.sequence_name}/{args.run_name}/figures'
    if not os.path.exists(f'{path_run_fig}'):
        os.makedirs(f'{path_run_fig}')
    plt.savefig(f'{path_run_fig}/epoch{epoch}.png')
    # plt.draw()
    # plt.show()
    # plt.pause(0.1)

# def make_figures_emotion(x, y, y_prim, class_names, epoch, metrics, args)::


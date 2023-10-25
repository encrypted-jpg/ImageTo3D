import random
from torch.autograd import Variable
import torch
import open3d as o3d
import matplotlib.pyplot as plt


def o3d_visualize_pc(pc):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([point_cloud])


def plot_image_output_gt(filename, image, output_pcd, gt_pcd, img_title='Image', output_title='Output PCD', gt_title='Ground Truth PCD', suptitle='', pcd_size=0.5, cmap='Reds', zdir='y',
                         xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5)):
    fig = plt.figure(figsize=(9, 3 * 1.4))
    elev = 30
    azim = -45

    # Plot the image
    ax_img = fig.add_subplot(1, 3, 1)
    ax_img.imshow(image)
    ax_img.set_title(img_title)

    # Plot the output point cloud
    color_output = output_pcd[:, 0]
    ax_output = fig.add_subplot(1, 3, 2, projection='3d')
    ax_output.view_init(elev, azim)
    ax_output.scatter(output_pcd[:, 0], output_pcd[:, 1], output_pcd[:, 2], zdir=zdir,
                      c=color_output, s=pcd_size, cmap=cmap, vmin=-1.0, vmax=0.5)
    ax_output.set_title(output_title)
    ax_output.set_axis_off()
    ax_output.set_xlim(xlim)
    ax_output.set_ylim(ylim)
    ax_output.set_zlim(zlim)

    # Plot the ground truth point cloud
    color_gt = gt_pcd[:, 0]
    ax_gt = fig.add_subplot(1, 3, 3, projection='3d')
    ax_gt.view_init(elev, azim)
    ax_gt.scatter(gt_pcd[:, 0], gt_pcd[:, 1], gt_pcd[:, 2], zdir=zdir,
                  c=color_gt, s=pcd_size, cmap=cmap, vmin=-1.0, vmax=0.5)
    ax_gt.set_title(gt_title)
    ax_gt.set_axis_off()
    ax_gt.set_xlim(xlim)
    ax_gt.set_ylim(ylim)
    ax_gt.set_zlim(zlim)

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)


def plot_pcd_one_view(filename, pcds, titles, suptitle='', sizes=None, cmap='Reds', zdir='y',
                      xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5)):
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3 * 1.4, 3 * 1.4))
    elev = 30
    azim = -45
    for j, (pcd, size) in enumerate(zip(pcds, sizes)):
        color = pcd[:, 0]
        ax = fig.add_subplot(1, len(pcds), j + 1, projection='3d')
        ax.view_init(elev, azim)
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir,
                   c=color, s=size, cmap=cmap, vmin=-1.0, vmax=0.5)
        ax.set_title(titles[j])
        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (
            max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) >
                0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

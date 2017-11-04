from TripleGAN import TripleGAN

from utils import show_all_variables
from utils import check_folder

import tensorflow as tf
import argparse

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of TripleGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--n', type=int, default=4000, help='The number of dataset')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'fashion-mnist', 'celebA', 'cifar10'],
                        help='The name of dataset')
    # In now, only cifar 10...
    parser.add_argument('--epoch', type=int, default=1000, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=20, help='The size of batch')
    parser.add_argument('--unlabel_batch_size', type=int, default=250, help='The size of unlabel batch')
    parser.add_argument('--z_dim', type=int, default=100, help='Dimension of noise vector')
    parser.add_argument('--gan_lr', type=float, default=2e-4, help='learning rate of GAN')
    parser.add_argument('--cla_lr', type=float, default=2e-3, help='learning rate of Classify')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    # --z_dim
    try:
        assert args.z_dim >= 1
    except:
        print('dimension of noise vector must be larger than or equal to one')

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = TripleGAN(sess, epoch=args.epoch, batch_size=args.batch_size, unlabel_batch_size=args.unlabel_batch_size,
                        z_dim=args.z_dim, dataset_name=args.dataset, n=args.n, gan_lr = args.gan_lr, cla_lr = args.cla_lr,
                        checkpoint_dir=args.checkpoint_dir, result_dir=args.result_dir, log_dir=args.log_dir)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        # launch the graph in a session
        gan.train()
        print(" [*] Training finished!")

        # visualize learned generator
        gan.visualize_results(args.epoch-1)
        print(" [*] Testing finished!")

if __name__ == '__main__':
    main()
import argparse

parser = argparse.ArgumentParser(description='Parser for all the training options')

# General options
parser.add_argument('-shuffle', action='store_true', help='Reshuffle data at each epoch')
parser.add_argument('-small_set', action='store_true', help='Whether uses a small dataset')
parser.add_argument('-train_record', action='store_true', help='Path to save train record')
parser.add_argument('-test_only', action='store_true', help='Only conduct test on the validation set')
parser.add_argument('-ckpt', default=0, type=int, help='Choose the checkpoint to run')

parser.add_argument('-model', required=True, help='Model type when we create a new one')
parser.add_argument('-data_dir', required=True, help='Path to data directory')
parser.add_argument('-train_list', required=True, help='Path to data directory')
parser.add_argument('-test_list', required=True, help='Path to data directory')
parser.add_argument('-save_path', required=True, help='Path to save train record')
parser.add_argument('-output_classes', required=True, type=int, help='Num of color classes')

# Training options
parser.add_argument('-learn_rate', default=1e-2, type=float, help='Base learning rate of training')
parser.add_argument('-momentum', default=0, type=float, help='Momentum for training')
parser.add_argument('-weight_decay', default=0, type=float, help='Weight decay for training')
parser.add_argument('-n_epochs', default=20, type=int, help='Training epochs')
parser.add_argument('-batch_size', default=64, type=int, help='Size of mini-batches for each iteration')
parser.add_argument('-criterion', default='CrossEntropy', help='Type of objective function')

# Model options
parser.add_argument('-pretrained', default=None, help='Path to the pretrained model')
parser.add_argument('-resume', action='store_true', help='Whether continue to train from a previous checkpoint')
parser.add_argument('-nGPU', default=1, type=int, help='Number of GPUs for training')
parser.add_argument('-workers', default=2, type=int, help='Number of subprocesses to to load data')
parser.add_argument('-decay', default=8, type=int, help='LR decay')

parser.add_argument('-size', default=100, type=int)

parser.add_argument('-cutout', action='store_true', default=False, help='apply cutout')
parser.add_argument('-n_holes', type=int, default=1, help='number of holes to cut out from image')
parser.add_argument('-length', type=int, default=8,
                    help='length of the holes')

args = parser.parse_args()

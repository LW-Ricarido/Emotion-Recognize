import os
import torch
import numpy as np


class Logger:
    def __init__(self, args, state):
        if not state:
            self.state = dict()
            self.state['epoch'] = 0
            self.state['best_acc'] = 0
        else:
            self.state = state

        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)

        self.save_path = os.path.join(args.save_path, args.model)

        if args.small_set:
            self.save_path += '-Small'
        else:
            self.save_path += '-Baseline'

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        if args.train_record and not args.test_only:
            self.train_record = []
        else:
            self.train_record = None

    def record(self, epoch, train_summary=None, test_summary=None, model=None):
        assert train_summary != None or test_summary != None, 'Need at least one summary'

        if torch.typename(model).find('DataParallel') != -1:
            model = model.module

        self.state['epoch'] = epoch

        if train_summary:
            latest = os.path.join(self.save_path, 'latest.pth')
            torch.save({'latest': epoch}, latest)

            model_file = os.path.join(self.save_path, 'model_%d.pth' % epoch)

            checkpoint = dict()
            checkpoint['state'] = self.state
            checkpoint['model'] = model.state_dict()

            torch.save(checkpoint, model_file)

            keys = train_summary.keys()
            keys = sorted(keys)
            train_rec = [train_summary[key] for key in keys]

        if test_summary:
            top3 = test_summary['acc-top3']
            state_top3 = self.state['best_acc']

            if top3 > state_top3:
                self.state['best_acc'] = test_summary['acc-top3']
                self.state['best_epoch'] = epoch

                best = os.path.join(self.save_path, 'best.pth')
                torch.save({'best': epoch}, best)

            keys = test_summary.keys()
            keys = sorted(keys)
            test_rec = [test_summary[key] for key in keys]

        if self.train_record is not None:
            self.train_record.append(train_rec + test_rec)
            record = os.path.join(self.save_path, 'train.record')
            np.save(record, self.train_record)

    def final_print(self):
        print("- Best:  Acc %6.3f at %d" % (
            self.state['best_acc'],
            self.state['best_epoch']))

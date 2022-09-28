class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'plot18_2019':
            return 'F:/zhanghq/rgbd_segmentation/coral_segmentation/datasets/Plot_18-2019_448stride224_pocill/data'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
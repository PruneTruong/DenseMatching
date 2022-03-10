import importlib
import os
from collections import OrderedDict


def create_default_local_file():
    """ Contains the path to all necessary datasets or useful folders (like workspace, pretrained models..)"""
    path = os.path.join(os.path.dirname(__file__), 'local.py')

    empty_str = '\'\''
    default_settings = OrderedDict({
        'workspace_dir': empty_str,
        'tensorboard_dir': 'self.workspace_dir',
        'pretrained_networks': 'self.workspace_dir',
        'pre_trained_models_dir' : empty_str,
        'megadepth': empty_str,
        'megadepth_csv': empty_str,
        'robotcar': empty_str,
        'robotcar_csv': empty_str,
        'hp': empty_str,
        'eth3d': empty_str,
        'kitti2012': empty_str,
        'kitti2015': empty_str,
        'sintel': empty_str,
        'scannet_test': empty_str,
        'yfcc': empty_str,
        'tss': empty_str,
        'PFPascal': empty_str,
        'PFWillow': empty_str,
        'spair': empty_str,
        'caltech': empty_str,
        'training_cad_520': empty_str,
        'validation_cad_520': empty_str,
        'coco': empty_str,
        'megadepth_training': empty_str
        })

    comment = {'workspace_dir': 'Base directory for saving network checkpoints.',
               'tensorboard_dir': 'Directory for tensorboard files.'}

    with open(path, 'w') as f:
        f.write('class EnvironmentSettings:\n')
        f.write('    def __init__(self):\n')

        for attr, attr_val in default_settings.items():
            comment_str = None
            if attr in comment:
                comment_str = comment[attr]
            if comment_str is None:
                f.write('        self.{} = {}\n'.format(attr, attr_val))
            else:
                f.write('        self.{} = {}    # {}\n'.format(attr, attr_val, comment_str))


def env_settings():
    env_module_name = 'admin.local'
    try:
        env_module = importlib.import_module(env_module_name)
        return env_module.EnvironmentSettings()
    except:
        env_file = os.path.join(os.path.dirname(__file__), 'local.py')

        create_default_local_file()
        raise RuntimeError('YOU HAVE NOT SETUP YOUR local.py!!!\n Go to "{}" and set all the paths you need. '
                           'Then try to run again.'.format(env_file))

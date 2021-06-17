import subprocess
import os


def prepare_data(root_path, mode='default'):
    """Untar data at the specified root_path. Only useful for euler cluster"""
    if mode == 'euler':
        tar_file_path = root_path

        tar_file_name = os.path.split(tar_file_path)[-1].split('.')[0]
        untar_path = '{}'.format(os.environ["TMPDIR"])
        out_path = '{}/{}'.format(os.environ["TMPDIR"], tar_file_name)

        marker_path = '{}_done.txt'.format(out_path)
        if not os.path.isfile(marker_path):
            if tar_file_path.endswith('.tar.gz'):
                cmd = 'tar -I pigz -xf {} -C {}'.format(tar_file_path, untar_path)
            elif tar_file_path.endswith('tar'):
                cmd = 'tar -xf {} -C {}'.format(tar_file_path, untar_path)
            elif tar_file_path.endswith('tar.xz'):
                cmd = 'tar -xvf {} -C {}'.format(tar_file_path, untar_path)
            elif tar_file_path.endswith('.zip'):
                cmd = 'unzip {} -d {}'.format(tar_file_path, untar_path)
            elif os.path.isdir(tar_file_path):
                # directory containing multiple tar files
                cmd = 'bash get_dataset.sh -n 16 -d {}'.format(tar_file_path)
            else:
                raise ValueError('Untaring file selected not valid : {}'.format(root_path))
            print('Copying data: {}'.format(cmd))
            out = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
            stdout, stderr = out.communicate()
            print(stdout)
            print(stderr)
            with open(marker_path, mode='a'):
                pass


def dataset_wrapper(base_dataset, mode='default'):
    if mode == 'default':
        base_dataset.initialize()
    elif mode == 'euler':
        if hasattr(base_dataset, 'prepare_data_euler'):
            base_dataset.prepare_data_euler()
        else:
            base_dataset = prepare_data_euler(base_dataset)
    else:
        raise Exception('Unknown mode {}'.format(mode))

    return base_dataset


def prepare_data_euler(base_dataset):
    root_path = base_dataset.root
    tar_file_path = base_dataset.root

    tar_file_name = os.path.split(tar_file_path)[-1].split('.')[0]
    print(tar_file_name)
    untar_path = '{}'.format(os.environ["TMPDIR"])
    out_path = '{}/{}'.format(os.environ["TMPDIR"], tar_file_name)

    marker_path = '{}_done.txt'.format(out_path)
    if not os.path.isfile(marker_path):
        cmd = 'tar xvf {} -C {}'.format(tar_file_path, untar_path)
        print('Copying data: {}'.format(cmd))
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        stdout, stderr = out.communicate()
        print(stdout)
        print(stderr)
        with open(marker_path, mode='a'):
            pass

    base_dataset.root = out_path
    # base_dataset.initialize()

    return base_dataset
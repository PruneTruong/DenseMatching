

class StatValue:
    def __init__(self):
        self.clear()

    def reset(self):
        self.val = 0

    def clear(self):
        self.reset()
        self.history = []

    def update(self, val):
        self.val = val
        self.history.append(self.val)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.clear()
        self.has_new_data = False

    def reset(self):
        self.avg = 0
        self.val = 0
        self.sum = 0
        self.count = 0

    def clear(self):
        self.reset()
        self.history = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def new_epoch(self):
        if self.count > 0:
            self.history.append(self.avg)
            self.reset()
            self.has_new_data = True
        else:
            self.has_new_data = False


def merge_dictionaries(list_dict, name=None):
    """Merges multiple dictionaries and add a specified suffix (listed in 'name') in front of the keys of
    each dictionary. """
    if name is not None:
        dall = {}
        for d, name_ in zip(list_dict, name):
            if name_ is None:
                dall.update(d)
            elif name_ == '':
                dall.update(d)
            else:
                for key in list(d.keys()):
                    dall['{}_'.format(name_) + key] = d[key]
    else:
        dall = {}
        for d in list_dict:
            dall.update(d)
    return dall


class Namespace:
    def __init__(self, dict_):
        self.__dict__.update(dict_)

    def update(self, dict_):
        self.__dict__.update(dict_)
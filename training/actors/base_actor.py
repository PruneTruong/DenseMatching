

class BaseActor:
    """ Base class for actor. The actor class handles the passing of the data through the network
    and calculation the loss"""
    def __init__(self, net, objective, batch_processing):
        """
        args:
            net - The network to train
            objective - The loss function
            batch_processing - A processing class which performs the necessary processing of the batched data.
        """
        self.net = net
        self.objective = objective
        self.batch_processing = batch_processing

    def __call__(self, mini_batch, training: bool):
        """ Called in each training iteration. Should pass in input data through the network, calculate the loss, and
        return the training stats for the input data
        args:
            mini_batch - A TensorDict containing all the necessary data blocks.
            training   - Bool indicating if training or evaluation mode
        returns:
            loss    - loss for the input data
            stats   - a dict containing detailed losses
        """
        raise NotImplementedError

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.net.to(device)

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)

    def eval(self):
        """ Set network to eval mode"""
        self.train(False)

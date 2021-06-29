from ..base_sampler import BaseSampler
import hashlib
from .random import getIndicesfromRandom
import random

class Random(BaseSampler):
    def __init__(self, data_list, sampling_params, images, descriptor_setup_hash, save=False):
        super().__init__()
        self.data_list = data_list
        self.sampling_method = "random"
        self.sampling_params = sampling_params
        self.images = images
        self.descriptor_setup_hash = descriptor_setup_hash
        self.save = save

        self.get_images_hash()
        self.get_sampler_setup_hash()

        self.load_or_sample()
        if save:
            self.save_sampled_indices()


    def sampling_procedure(self):
        # params for random sampler, length to be selected
        param = self.sampling_params.get("length")
        total_length = len(self.data_list)
        print("Before sampling: {}".format(total_length))
        if total_length == param:
            self.image_indices = list(range(0, total_length))
            # do not udpate data_list
        else:
            # if registered as a fraction, then default to a fraction of 
            # total training data
            if param < 1:
                length = int(total_length * param)
            else:
                length = param
            image_indices = getIndicesfromRandom(length, total_length)
            self.image_indices = image_indices
            self.update_data_list()

        print("After sampling: {}".format(len(self.data_list)))

    def get_sampler_setup_hash(self):
        string = ""
        sampler_setup = [self.sampling_params.get("length")]
        for num in sampler_setup:
            string += "%.15f" % num
        md5 = hashlib.md5(string.encode("utf-8"))
        hash_result = md5.hexdigest()
        self.sampler_setup_hash = hash_result
    
    def save_config(self):
        pass
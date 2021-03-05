from ..base_sampler import BaseSampler
from ..subsampling import subsampling_with_PCA

import hashlib
import numpy as np


class NearestNeighbor(BaseSampler):
    def __init__(self, data_list, sampling_params, images, descriptor_setup_hash, save=False):
        super().__init__()
        self.data_list = data_list
        self.sampling_method = "nns"
        self.sampling_params = sampling_params
        self.images = images
        self.descriptor_setup_hash = descriptor_setup_hash
        self.save = save

        self.cutoff = self.sampling_params.get("cutoff")
        self.rate = self.sampling_params.get("rate")
        self.method = self.sampling_params.get("method", "pykdtree")
        self.target_variance = self.sampling_params.get("target_variance", 0.999999)

        self.get_images_hash()
        self.get_sampler_setup_hash()

        self.load_or_sample()
        if save:
            self.save_sampled_indices()

    def sampling_procedure(self):
        self.prepare_descriptors()

        total_length = len(self.data_list)
        print("Before sampling: {} images.".format(total_length))

        _, image_indices = subsampling_with_PCA(
            self.fps_array,
            image_index=self.index_list, 
            cutoff_sig=self.cutoff,
            rate = self.rate,
            target_variance=self.target_variance,
            method = self.method, 
        )
        
        self.image_indices = list(set(image_indices))
        self.update_data_list()

        print("After sampling: {} images.".format(len(self.data_list)))
    
    def get_sampler_setup_hash(self):
        string = ""
        sampler_setup = [
            self.cutoff,
            self.rate,
            self.method
            ]
        for _ in sampler_setup:
            if type(_) is not str:
                string += "%.15f" % _
            else:
                string += _
        
        md5 = hashlib.md5(string.encode("utf-8"))
        hash_result = md5.hexdigest()
        self.sampler_setup_hash = hash_result

    def prepare_descriptors(self):
        fps_reorganized = []
        index_list = []
        for i, _data in enumerate(self.data_list):
            fp_image = _data.fingerprint.tolist()
            for fp in fp_image:
                fps_reorganized.append(fp)
                index_list.append(i)
        
        self.fps_array = np.asarray(fps_reorganized)
        self.index_list = index_list
    
        

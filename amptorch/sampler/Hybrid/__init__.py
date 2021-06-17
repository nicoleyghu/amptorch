from ..base_sampler import BaseSampler
from ..subsampling import subsampling_with_PCA
from ..Random.random import getIndicesfromRandom

import hashlib
import numpy as np
import json
import os
import time

class NNSRSHybrid(BaseSampler):
    def __init__(self, data_list, sampling_params, images, descriptor_setup_hash, save=False):
        super().__init__()
        self.data_list = data_list
        self.sampling_method = "hybrid"
        self.sampling_params = sampling_params
        self.images = images
        self.descriptor_setup_hash = descriptor_setup_hash
        self.save = save

        self.cutoff = self.sampling_params.get("cutoff")
        self.rate = self.sampling_params.get("rate")
        self.method = self.sampling_params.get("method", "pykdtree")
        self.start_trial_component = self.sampling_params.get("start_trial_component", 10)
        self.max_component = self.sampling_params.get("max_component", 30)
        self.target_variance = self.sampling_params.get("target_variance", 0.999999)
        self.random_ratio = self.sampling_params.get("random_ratio", 0.5)

        self.get_images_hash()
        self.get_sampler_setup_hash()

        self.load_or_sample()
        if save:
            self.save_sampled_indices()

    def sampling_procedure(self):
        self.prepare_descriptors()

        total_length = len(self.data_list)
        print("Before sampling: {} images.".format(total_length))

        t0 = time.time()
        # NNS sampling
        dict_results = subsampling_with_PCA(
            self.fps_array,
            image_index=self.index_list, 
            cutoff_sig=self.cutoff,
            rate = self.rate,
            start_trial_component=self.start_trial_component,
            max_component=self.max_component,
            target_variance=self.target_variance,
            method = self.method,
            verbose=2
        )
        
        nns_indices = list(set(dict_results["image_index_result"]))
        self.dict_results = dict_results

        # random sampling to add on more points
        length = int(len(nns_indices) / (1 - self.random_ratio)) # length for total results
        if length > total_length:
            self.image_indices = list(range(0, total_length))
        else: 
            rs_indices = getIndicesfromRandom(length, total_length, existing_list_of_indices=nns_indices)
            self.image_indices = rs_indices

        self.sampling_time = time.time() - t0
        self.update_data_list()

        print("After sampling: {} images.".format(len(self.data_list)))
    
    def get_sampler_setup_hash(self):
        string = ""
        sampler_setup = [
            self.cutoff,
            self.rate,
            self.start_trial_component,
            self.max_component,
            self.target_variance,
            self.method,
            self.random_ratio,
            ]
        for _ in sampler_setup:
            if type(_) is not str:
                string += "%.15f" % _
            else:
                string += _
        
        md5 = hashlib.md5(string.encode("utf-8"))
        hash_result = md5.hexdigest()
        self.sampler_setup_hash = hash_result
    
    def save_config(self):
        filename = self.sampled_indices_directory + "/" + self.sampler_setup_hash + ".json"

        # if dir not exist, make dir
        if os.path.isdir(self.sampled_indices_directory) is False:
            os.makedirs(self.sampled_indices_directory)

        config = {"sampling_params": {
                    "cutoff": self.cutoff,
                    "rate": self.rate,
                    "method": self.method,
                    "start_trial_component": self.start_trial_component,
                    "max_component": self.max_component,
                    "target_variance": self.target_variance, 
                    "random_ratio": self.random_ratio, 
                },
                "PCA_results":
                {
                    "num_PC_kept": self.dict_results["num_PC_kept"],
                    "explained_variance": self.dict_results["explained_variance"]
                },
                "sampling_results":
                {
                    "idnex_file": self.sampled_indices_directory + "/" + self.sampler_setup_hash + ".pkl",
                    "sampled_images": len(self.image_indices),
                    "sampling_time": self.sampling_time
                }
                }

        # save file
        json.dump(config, open(filename, "w+"), indent=2)

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
    
        

from abc import ABC, abstractmethod
import hashlib
import os
import pickle

class BaseSampler(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def sampling_procedure(self):
        pass

    @abstractmethod
    def get_sampler_setup_hash(self):
        pass

    def update_data_list(self):
        self.data_list = [self.data_list[_] for _ in self.image_indices]

    def save_sampled_indices(self):
        filename = self.sampled_indices_directory + "/" + self.sampler_setup_hash
        
        # if dir not exist, make dir
        if os.path.isdir(self.sampled_indices_directory) is False:
            os.makedirs(self.sampled_indices_directory)
        # save file
        with open(filename, "wb") as out_file:
            pickle.dump(self.image_indices, out_file)
            

    def load_or_sample(self):
        # filename = "processed/samplers/[images_hash]/[descriptor_setup_hash]/[sampler_setup_hash]"

        _dir = "/".join([self.sampling_method, self.images_hash])
        self.sampled_indices_directory = "./processed/samplers/" + _dir
        filename = self.sampled_indices_directory  + "/" + self.sampler_setup_hash

        # check for file existence and 
        if os.path.isfile(filename): 
            print("Loading sampled indices...")
            with open(filename, "rb") as in_file:
                self.image_indices = pickle.load(in_file)
                self.update_data_list()
        else: 
            print("Sampling...")
            self.sampling_procedure()

    def get_images_hash(self):
        string = str(self.images)
        md5 = hashlib.md5(string.encode("utf-8"))
        hash_result = md5.hexdigest()
        self.images_hash = hash_result

    def run(self):
        return self.data_list
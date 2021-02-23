import random

def construct_sampler(data_list, sampling_setup):
    sampling_method = sampling_setup.get("sampling_method", None)
    sampling_params = sampling_setup.get("sampling_params", None)
    image_idx = None
    if sampling_method == "random":
        image_idx = RandomSampler(data_list, sampling_params)
    elif sampling_method == "nns":
        image_idx = NearestNeighborSampler(data_list, sampling_params)
    else:
        raise NotImplementedError
    return image_idx

def RandomSampler(data_list, sampling_params):
    # params for random sampler, length to be selected
    param = sampling_params.get("length")
    # if registered as a fraction, then default to a fraction of 
    # total training data
    if param < 1:
        length = int(len(data_list) * param)
    else:
        length = param
    image_idx = random.sample(range(0, len(data_list)), length)

    return image_idx


def NearestNeighborSampler(data_list, sampling_params):
    return
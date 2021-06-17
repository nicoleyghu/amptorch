import random

def getIndicesfromRandom(length, total_length, existing_list_of_indices=None):
    if existing_list_of_indices is not None:
        image_indices = random.sample(range(0, total_length), length - len(existing_list_of_indices))
        image_indices.extend(existing_list_of_indices)
    else:
        image_indices = random.sample(range(0, total_length), length)
    image_indices = list(set(image_indices))
    while len(image_indices) < length:
        image_indices.append(random.randint(0, total_length))
        image_indices = list(set(image_indices))
    return image_indices
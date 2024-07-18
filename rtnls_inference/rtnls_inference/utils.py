import torch


def test_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return torch.tensor([])  # Return an empty tensor if all items are faulty
    # Convert batch list to a tensor or any suitable format for your model
    # This depends on the structure of your data items
    return torch.utils.data.dataloader.default_collate(batch)


def get_all_subclasses_dict(cls):
    all_subclasses_dict = {}

    for subclass in cls.__subclasses__():
        all_subclasses_dict[subclass.__name__] = subclass
        all_subclasses_dict.update(get_all_subclasses_dict(subclass))

    return all_subclasses_dict


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def decollate_batch(batch):
    """
    Separate batched PyTorch tensors in a nested dictionary into individual items and convert them to numpy or primitive types if the size is 1.

    Args:
        batch (dict): A dictionary where each key has a tensor value batched along the first dimension, lists, or nested dictionaries.

    Returns:
        list: A list of dictionaries, where each dictionary represents an item from the original batch.
    """
    # Number of items in the batch, assuming all tensors have the same batch size
    batch_size = len(batch["image"])

    def convert(val):
        if isinstance(val, torch.Tensor):
            decollated_val = val.detach().cpu().numpy()
            if decollated_val.size == 1:
                return decollated_val.item()
            return decollated_val
        elif isinstance(val, dict):
            return decollate_batch(val)
        elif isinstance(val, list):
            return [convert(item) for item in val]
        else:
            return val

    # Recursive function to decollate nested dictionaries and lists
    def recursive_decollate(batch, index):
        if isinstance(batch, dict):
            return {
                key: recursive_decollate(value, index) for key, value in batch.items()
            }
        elif isinstance(batch, list):
            return convert(batch[index])
        elif isinstance(batch, torch.Tensor):
            return convert(batch[index])
        else:
            return batch

    # Decollate the batch
    decollated = [recursive_decollate(batch, i) for i in range(batch_size)]

    return decollated

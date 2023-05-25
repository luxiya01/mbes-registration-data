import open3d as o3d
import numpy as np
from functools import partial
import torch
from lib.timer import Timer
from lib.utils import load_obj, natural_key
from datasets import mbes_data 


def calibrate_neighbors(dataset, config, collate_fn, keep_ratio=0.8, samples_threshold=2000):
    timer = Timer()
    last_display = timer.total_time

    # From config parameter, compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (config.deform_radius + 1) ** 3))
    neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        timer.tic()
        batched_input = collate_fn([dataset[i]], config, neighborhood_limits=[hist_n] * 5)

        # update histogram
        counts = [torch.sum(neighb_mat < neighb_mat.shape[0], dim=1).numpy() for neighb_mat in batched_input['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighb_hists += np.vstack(hists)
        timer.toc()

        if timer.total_time - last_display > 0.1:
            last_display = timer.total_time
            print(f"Calib Neighbors {i:08d}: timings {timer.total_time:4.2f}s")

        if np.min(np.sum(neighb_hists, axis=1)) > samples_threshold:
            break

    cumsum = np.cumsum(neighb_hists.T, axis=0)
    percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]), axis=0)

    neighborhood_limits = percentiles
    print('\n')

    return neighborhood_limits

def get_datasets(config):
    if(config.dataset=='multibeam'):
        train_set, val_set = mbes_data.get_multibeam_train_datasets(config)
        benchmark_set = mbes_data.get_multibeam_test_datasets(config)
    else:
        raise NotImplementedError

    return train_set, val_set, benchmark_set



def get_dataloader(dataset, batch_size=1, num_workers=4, shuffle=True, neighborhood_limits=None):
    if neighborhood_limits is None:
        neighborhood_limits = calibrate_neighbors(dataset, dataset.config, collate_fn=collate_fn_descriptor)
    print("neighborhood:", neighborhood_limits)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # https://discuss.pytorch.org/t/supplying-arguments-to-collate-fn/25754/4
        collate_fn=partial(collate_fn_descriptor, config=dataset.config, neighborhood_limits=neighborhood_limits),
        drop_last=False
    )
    return dataloader, neighborhood_limits


if __name__ == '__main__':
    pass

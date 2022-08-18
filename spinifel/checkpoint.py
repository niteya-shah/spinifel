import pickle
from pathlib import Path
import os

def generate_checkpoint_name(outdir: str, gen_num: int, tag='') -> str:
    """Constructs and returns full path of a pickle filename"""
    fname = f'generation_{gen_num}'
    if tag:
        fname += f'_{tag}'
    fname += '.pickle'
    return os.path.join(outdir, fname)

def save_checkpoint(
        res: dict,
        outdir: str,
        gen_num: int,
        tag='',
        protocol=4) -> None:
    """Save results as pickle
       protocol=4 is the highest supported on psana nodes
    """
    # create outdir if does not exist
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # save pickle
    fname = generate_checkpoint_name(outdir, gen_num, tag)
    with open(fname, 'wb') as handle:
        pickle.dump(res, handle, protocol=protocol)


def load_checkpoint(outdir: str, gen_num: int, tag='') -> dict:
    """Load results as pickle"""
    fname = generate_checkpoint_name(outdir, gen_num, tag)
    with open(fname, 'rb') as handle:
        return pickle.load(handle)

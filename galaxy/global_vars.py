import argparse

_GLOBAL_ARGS = None

def initial_args():
    global _GLOBAL_ARGS

    parser = argparse.ArgumentParser(description='rank: device id')
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--world', default=1, type=int)
    _GLOBAL_ARGS = parser.parse_args()

def get_args():
    """Return arguments."""
    assert (
    _GLOBAL_ARGS is not None
    ), 'global arguments is not initialized'
    
    return _GLOBAL_ARGS
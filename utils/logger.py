from datetime import datetime
from pathlib import Path
import yaml


def mkdirs(paths):
    if isinstance(paths, (str, Path)):
        Path(paths).mkdir(parents=True, exist_ok=True)
    else:
        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)


def parse(args):
    with open(args.config, 'r') as f:
        opt = yaml.safe_load(f)

    # Set up experiment directory
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    experiments_root = Path('experiments') / f"{opt['name']}_{timestamp}"
    opt['path']['experiments_root'] = str(experiments_root)
    opt['path']['checkpoint'] = str(experiments_root / 'checkpoint')
    mkdirs(opt['path']['checkpoint'])

    opt['distributed'] = False
    return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None


def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        return NoneDict({k: dict_to_nonedict(v) for k, v in opt.items()})
    elif isinstance(opt, list):
        return [dict_to_nonedict(x) for x in opt]
    return opt

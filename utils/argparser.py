import argparse
import os
import json

from utils.path_utils import expand_data_root_strings


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of MetaVLM')

    # Add arguments to accept multiple config files
    parser.add_argument('--config', type=str, action='append', help='Path to configuration file(s)', required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    return config


def load_config(config_files):
    def update_config(base_config, new_config):
        for key, value in new_config.items():
            if isinstance(value, dict) and key in base_config:
                update_config(base_config[key], value)
            else:
                base_config[key] = value

    config = {}
    for config_file in config_files:
        assert os.path.exists(config_file), f"Configuration file {config_file} does not exist."
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            update_config(config, config_data)
    return expand_data_root_strings(config)
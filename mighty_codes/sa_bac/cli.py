"""Command-line tool functionality for sa-bac."""

from ruamel_yaml import YAML
import logging
import os
import sys
from datetime import datetime

from mighty_codes.cli.base_cli import AbstractCLI
from mighty_codes.sa_bac.main import SimulatedAnnealingBinaryAsymmetricChannel

yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)


class CLI(AbstractCLI):
    """CLI implements AbstractCLI from the mighty_codes package."""

    def __init__(self):
        self.name = 'sa-bac'
        self.args = None

    def get_name(self) -> str:
        return self.name

    def validate_args(self, args):
        """Validate parsed arguments."""

        # Ensure that if there's a tilde for $HOME in the file path, it works.
        try:
            args.input_yaml_file = os.path.expanduser(args.input_yaml_file)
        except TypeError:
            raise ValueError("Problem with provided input paths.")

        self.args = args

        return args

    def run(self, args):
        """Run the main tool functionality on parsed arguments."""

        try:
            with open(args.input_yaml_file, 'r') as f:
                params = yaml.load(f)
        except IOError:
            raise RuntimeError(f"Error loading the input YAML file {args.input_yaml_file}!")

        assert 'output_root' in params
        assert os.access(params['output_root'], os.W_OK)
        
        # Send logging messages to stdout as well as a log file.
        log_file = os.path.join(params['output_root'], "run_sa_bac.log")
        logging.basicConfig(
            level=logging.INFO,
            format="mighty-codes:sa-bac:%(asctime)s: %(message)s",
            filename=log_file,
            filemode="w")
        console = logging.StreamHandler()
        formatter = logging.Formatter("mighty-codes:sa-bac:%(asctime)s: %(message)s", "%H:%M:%S")
        console.setFormatter(formatter)  # Use the same format for stdout.
        logging.getLogger('').addHandler(console)  # Log to stdout and a file.

        # Log the command as typed by user.
        logging.info("Command:\n" + ' '.join(['mighty-codes', 'sa-bac'] + sys.argv[2:]))
                                      
        # instantiate
        sabac = SimulatedAnnealingBinaryAsymmetricChannel(
            params=params,
            logger=logging.info)
        
        # run!
        sabac.run()

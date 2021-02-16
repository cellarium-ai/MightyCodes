"""Command-line tool functionality for yaml-tools."""

from mighty_codes.cli.base_cli import AbstractCLI

from ruamel_yaml import YAML
import logging
import os
import sys
from datetime import datetime
from typing import Any

yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)


class CLI(AbstractCLI):
    """CLI implements AbstractCLI from the mighty_codes package."""

    def __init__(self):
        self.name = 'yaml-tools'
        self.args = None

    def get_name(self) -> str:
        return self.name

    def validate_args(self, args):
        """Validate parsed arguments."""

        # Ensure that if there's a tilde for $HOME in the file path, it works.
        try:
            args.input_yaml_file = os.path.expanduser(args.input_yaml_file)
        except TypeError:
            raise ValueError("Problem with provided input YAML file.")

        try:
            args.output_yaml_file = os.path.expanduser(args.output_yaml_file)
        except TypeError:
            raise ValueError("Problem with provided output YAML file.")

        self.args = args

        return args

    def run(self, args):
        """Run the main tool functionality on parsed arguments."""

        try:
            with open(args.input_yaml_file, 'r') as f:
                params = yaml.load(f)
        except IOError:
            raise RuntimeError(f"Error loading the input YAML file {args.input_yaml_file}!")
        
        # Send logging messages to stdout as well as a log file.
        log_file = os.path.join(os.getcwd(), "run_yaml_tools.log")
        logging.basicConfig(
            level=logging.INFO,
            format="mighty-codes:yaml-tools:%(asctime)s: %(message)s",
            filename=log_file,
            filemode="w")
        console = logging.StreamHandler()
        formatter = logging.Formatter("mighty-codes:yaml-tools:%(asctime)s: %(message)s", "%H:%M:%S")
        console.setFormatter(formatter)  # Use the same format for stdout.
        logging.getLogger('').addHandler(console)  # Log to stdout and a file.

        # Log the command as typed by user.
        logging.info("Command:\n" + ' '.join(['mighty-codes', 'yaml-tools'] + sys.argv[2:]))

        if args.update is None:
            logging.info("No update specified -- saving the YAML file as is.")
        else:
            for elem in args.update:
                assert len(elem) == 3, "Bad update arguments; expected format: <key> <value> <type>"
                logging.info(f"Parsing YAML update {elem} ...")
                key = str(elem[0])
                str_value = str(elem[1])
                str_type = str(elem[2])
                value = self.cast(str_value, str_type)
                if key in params:
                    logging.info(f"Key {key} already exists; old value: {params[key]}, new value: {value}")
                    params.yaml_set_comment_before_after_key(
                        key,
                        before=f"Updated by yaml-tools; old value: {params[key]}, new value: {value}")                        
                    params[key] = value
                else:
                    logging.info(f"Key {key} does not exist; assigned value: {value}")
                    params.yaml_set_comment_before_after_key(
                        key,
                        before=f"\nAdded by yaml-tools")
                    params[key] = value
        
        # save YAML file
        logging.info(f"Saving updated YAML file to {args.output_yaml_file} ...")
        with open(args.output_yaml_file, 'w') as f:
            yaml.dump(params, f)
            
        logging.info(f"Done!")

    def cast(self, str_value: str, str_type: str) -> Any:
        if str_type == 'int':
            try:
                value = int(str_value)
            except ValueError:
                raise ValueError(f"Could not cast {str_value} to int!")
        elif str_type == 'float':
            try:
                value = float(str_value)
            except ValueError:
                raise ValueError(f"Could not cast {str_value} to float!")
        elif str_type == 'str':
            value = str_value
        else:
            raise ValueError(f"Unknown type {str_type}! allowed values: int, float, str")
        return value

import argparse


def add_subparser_args(subparsers: argparse) -> argparse:
    """Add tool-specific arguments.

    Args:
        subparsers: Parser object before addition of arguments specific to
            sa-bac.

    Returns:
        parser: Parser object with additional parameters.

    """

    subparser = subparsers.add_parser(
        "yaml-tools",
        description="A tool for modifying and saving YAML files.",
        help="A tool for modifying and saving YAML files.")

    subparser.add_argument(
        "-i",
        "--input-yaml-file",
        nargs=None,
        type=str,
        dest='input_yaml_file',
        default=None,
        required=True,
        help="Input YAML file.")

    subparser.add_argument(
        "-o",
        "--output-yaml-file",
        nargs=None,
        type=str,
        dest='output_yaml_file',
        default=None,
        required=True,
        help="Output YAML file.")
    
    subparser.add_argument(
        "-u",
        "--update",
        nargs=3,
        action='append',
        default=None,
        required=False,
        help="Update key-value in YAML file. Expects three arguments: <key> <value> <type>.")

    return subparsers

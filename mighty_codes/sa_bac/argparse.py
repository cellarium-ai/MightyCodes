import argparse


def add_subparser_args(subparsers: argparse) -> argparse:
    """Add tool-specific arguments for remove-background.

    Args:
        subparsers: Parser object before addition of arguments specific to
            sa-bac.

    Returns:
        parser: Parser object with additional parameters.

    """

    subparser = subparsers.add_parser(
        "sa-bac",
        description="Code construction for BAC channels using parallel simulated annealing.",
        help="Code construction for BAC channels using parallel simulated annealing.")

    subparser.add_argument(
        "-i",
        "--input-params-yaml-file",
        nargs=None,
        type=str,
        dest='input_params_yaml_file',
        default=None,
        required=True,
        help="YAML file specifying run parameters.")

    return subparsers

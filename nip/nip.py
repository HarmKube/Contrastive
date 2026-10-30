import argparse
import json

from cli.cli_models import CliArgument, CliAction
from config.global_config import config

class ConfigEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return vars(obj)
        except TypeError:
            return str(obj)

def print_help(arguments: list[CliArgument], actions: list[CliAction]):
    triggers = []
    for action in actions:
        triggers.append(action.trigger_arg.replace('_', '-'))

    trigger_args = [arg for arg in arguments if arg.name in triggers]
    print("Help for NIP CLI:\n")
    print("Usage: nip.py [action] [options]\n")
    print("Actions:")
    for arg in trigger_args:
        # Format with consistent spacing between short/long options and description
        short_opt = f"-{arg.short}" if arg.short else ""
        long_opt = f"--{arg.name}"
        print(f"  {short_opt:<4} {long_opt:<25} {arg.help}")
    print("\nSee nip.py [action] -h, --help for more information about the action and its options.")

if __name__ == "__main__":
    # load the CLI configuration from the cli_config module
    from cli.cli_config import argument, actions

    # create the argument parser and add the arguments to it
    parser = argparse.ArgumentParser(description="NIP CLI", add_help=False)

    parser.add_argument('-h', '--help', action='store_true', help='Show this help message and exit')

    for arg in argument:
        arg.add_to_parser(parser)

    # parse the command line arguments
    args = parser.parse_args()

    # check if the help flag was set
    if args.help:
        for action in actions:
            if action.print_help(args, argument):
                break
        else:
            print_help(argument, actions)
        exit()

    if args.config:
        config.from_yaml(args.config)
    else:
        config.from_yaml('./config/default.yaml')
    
    print("\nLoaded Configuration:")
    print(json.dumps(config, indent=2, sort_keys=True, cls=ConfigEncoder))
    print()  # Add an empty line for better readability

    # check if any action should be started based on the parsed arguments
    for action in actions:
        if action.start_action(args):
            break
    else:
        print_help(argument, actions)

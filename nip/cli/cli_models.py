from argparse import ArgumentParser


class CliArgument:
    '''
    Class to define command line arguments for the CLI.
    Each argument can be a flag or a value.
    The class also provides a method to add the argument to an ArgumentParser instance.
    name: str - The name of the argument.
    is_flag: bool - Whether the argument is a flag (True) or a value (False).
    short: str - The short version of the argument (e.g., 'b' for 'benchmark').
    help: str - The help text for the argument.
    '''
    def __init__(self, name: str, is_flag: bool, short:str, help: str = None):
        self.name = name
        self.is_flag = is_flag
        self.short = short
        self.help = help

    def add_to_parser(self, parser:ArgumentParser):
        if self.is_flag:
            parser.add_argument(f'-{self.short}', f'--{self.name}', action='store_true', help=self.help)
        else:
            parser.add_argument(f'-{self.short}', f'--{self.name}', help=self.help)

class CliAction:
    '''
    Class to define actions that can be triggered by command line arguments.
    Each action is associated with a function that will be called when the action is triggered.
    name: str - The name of the action.
    trigger_arg: str - The name of the argument that triggers the action. Should a be flag.
    params_req: list[str] - List of required parameters for the action.
    params_opt: list[str] - List of optional parameters for the action.
    func: callable - The function to be called when the action is triggered.
    '''
    def __init__(self, name: str, trigger_arg:str, params_req: list[str],params_opt: list[str], func: callable):
        self.name = name
        self.trigger_arg = trigger_arg
        self.params_req = params_req
        self.params_opt = params_opt
        self.func = func

    def start_action(self, args):
        # Check if the action should be started based on the trigger argument
        if hasattr(args, self.trigger_arg) and getattr(args, self.trigger_arg):
            # Check if the required parameters are provided
            missing_params = [param for param in self.params_req if getattr(args, param, None) is None]
            if len(missing_params) > 0:
                print(f"Missing required parameters for {self.name}: {', '.join(missing_params)}")
                return True
            
            # parse the required and optional parameters
            parsed_params_req = {param: getattr(args, param, None) for param in self.params_req}            
            parsed_params_opt = {param: getattr(args, param, None) for param in self.params_opt if getattr(args, param, None) is not None}
            params = {**parsed_params_req, **parsed_params_opt}

            # Call the function associated with the action
            print(f"Running {self.name} with params: {params}")
            self.func(**params)
            print(f"Finished {self.name}")
            return True
        return False
    
    def print_help(self, args, arguments: list[CliArgument]):
        # Check if the help should be printed based on the trigger argument
        if hasattr(args, self.trigger_arg) and getattr(args, self.trigger_arg):
            trigger_arg = next((arg for arg in arguments if arg.name == self.trigger_arg.replace('_', '-')), None)
            # Filter the arguments to only include the required and optional arguments
            optional_args = [arg for arg in arguments if arg.name in self.params_opt]
            required_args = [arg for arg in arguments if arg.name in self.params_req]
            # Create the string for the required and optional arguments
            required_args_str = [f"-{arg.short:<4} --{arg.name:<10} {str.upper(arg.name) if not arg.is_flag else '':<10} {arg.help}" for arg in required_args]
            optional_args_str = [f"-{arg.short:<4} --{arg.name:<10} {str.upper(arg.name) if not arg.is_flag else '':<10} {arg.help}" for arg in optional_args]
            # Print the help message
            print(f"Help for {self.name}:\n")
            required_str = ' '.join([f'[-{arg.short} {str.upper(arg.name) if not arg.is_flag else ""}]' for arg in required_args])
            optional_str = ' '.join([f'[-{arg.short} {str.upper(arg.name) if not arg.is_flag else ""}]' for arg in optional_args])
            print(f"Usage: nip.py -{trigger_arg.short} {required_str} {optional_str}\n")
            print("required arguments:")
            for arg in required_args_str:
                print(f"  {arg}")
            print("\noptional arguments:")
            for arg in optional_args_str:
                print(f"  {arg}")
            return True
        return False

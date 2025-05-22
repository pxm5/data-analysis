## Write the command line interface here.
import click
import pandas as pd
from analysis import funcs
from analysis.funcs import *
from analysis import shared
import pyfiglet
from rich import print


def run():
    banner()
    path = input("Enter path to your csv file: \n")
    try:
        shared.df = pd.read_csv(path)
    except Exception as e:
        print(f"Failed to read the file: {e}")
        exit(1)
    print("[bold red]COMMANDS QUICKHELP:[/bold red]")
    print("exit: Exit the CLI tool")
    print("cmds: Shows all commands and methods available to you")
    while True:
        try:
            command = click.prompt(click.style(">>>", fg='blue', bold=True))
            if command.lower() =="exit":
                print("Thanks for using my tool, until next time!")
                break
            if command.lower() =="cmds":
                command_names()
                continue
            args = command.strip().split()
            cmd_name = args[0]
            if cmd_name in main.commands:
                cmd = main.commands[cmd_name]
                cmd(args[1:])
        except SystemExit:
            pass
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"ERROR: {e}")
    exit(0)

@click.group()
def main():
    pass
    
def banner():
    banner = pyfiglet.figlet_format("DATA CLI", font="eftitalic")
    formatted = click.style(banner, fg='green', bold=True)
    click.echo(formatted)

def command_names():
    x = 0
    for name, cmd in main.commands.items():
        if isinstance(cmd, click.core.Command):
            print(f'{x}. {name}')
            x+=1

for command in dir(funcs):
    cmd = getattr(funcs, command)
    if isinstance(cmd, click.core.Command):
        main.add_command(cmd, name=command)


if __name__=="__main__":
    run()
    
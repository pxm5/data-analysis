## Write the command line interface here.
import click
import pandas as pd
import funcs
from funcs import *
import shared

@click.group()
def main():
    pass
    

@main.command()
def show():
    click.echo(shared.df)

for command in dir(funcs):
    cmd = getattr(funcs, command)
    if isinstance(cmd, click.core.Command):
        main.add_command(cmd, name=command)

if __name__ == "__main__":
    path = input("Enter path to your csv file: ")
    try:
        shared.df = pd.read_csv(path)
    except Exception as e:
        print(f"Failed to read the file: {e}")
        exit(1)
    print("These are all of the commands available: ")
    x = 0
    for command in dir(funcs):
        check = getattr(funcs, command)
        if isinstance(check, click.core.Command):
            print(f"{x}. "+command)
            x+=1
    while True:
        try:
            command = input(">>> ")
            if command.lower() =="exit":
                print("Thanks for using my tool, until next time!")
                break
            args = command.strip().split()
            main(args, standalone_mode=False)
        except SystemExit:
            pass
        except Exception as e:
            print(f"Error caused. ERROR: {e}")

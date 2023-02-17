# ts-qa

Question Answering for Time Series 🚀

## Howto use
To run the project one can either build the vscode devcontainer or run a headless compose version. For this one can leverage the given makefile. 
Executing the make file requires a unix/linux environment.

- Running `make` builds the container and then attaches to the shell of the container
- One must manually stop and/or remove the running container. This can be done by running `make stop` and/or `make remove` or `make stop_remove`.

Also run `pip install -e .`.

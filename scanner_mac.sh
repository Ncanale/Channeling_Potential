#!/bin/zsh
# scanner.sh no longer uses GNU-only commands (sed -i, nproc), so it runs on macOS as is.
exec bash "$(dirname "$0")/scanner.sh" "$@"

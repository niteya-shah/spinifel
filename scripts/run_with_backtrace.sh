#!/bin/bash

set -e

"$@" &
pid=$!

sleep 3m

gdb -p $pid -ex 'set width 0' -ex 'thread apply all backtrace' -ex 'quit'

sleep 20

kill %1

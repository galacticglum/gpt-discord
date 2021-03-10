#!/bin/bash
#
# A script for running the chat bot given a persona username.
# Usage: ./run_chat_bot.sh <persona_username>

if [ '$0' != '' ]; then
    python chat_bot.py --persona-username $0 -c instance/mat137_sim_chat_bot_config.json
else
    echo 'Invalid arguments. See usage for documentation on arguments.'
fi
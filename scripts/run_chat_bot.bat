@echo off
REM A script for running the chat bot given a persona username.
REM Usage: ./run_chat_bot.bat <persona_username>
python chat_bot.py --persona-username %1 -c instance/mat137_sim_chat_bot_config.json
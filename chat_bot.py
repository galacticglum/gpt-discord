"""A discord chat bot that uses a GPT2 model for message generation."""
import re
import json
import random
import logging
import platform
import argparse
import collections
from pathlib import Path
from datetime import timedelta
from typing import Optional, Union, Deque, List, Set, Dict, Pattern

import asyncio
import discord
from transformers import PreTrainedModel, PreTrainedTokenizer

from gpt2_model import load_model, generate


def _set_selector_event_loop_policy() -> None:
    """Sets the asyncio event loop policy to SelectorEventLoop on Windows platform."""
    if platform.system() != 'Windows':
        return
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception as exception:
        print(f'Failed to set selector event loop policy! {exception}')


def _unescape_string(string: str) -> str:
    """Unescape a unicode string.
    """
    string = string.encode('utf-8').decode('unicode_escape')
    # Replace escaped quotes and smileys
    string = string.replace('\>', '>')
    string = string.replace('\:', ':')
    return string

class Client(discord.Client):
    """A discord.Client implementation for the chat bot.

    Instance Attributes:
        - channel_ids: The ids of the channels to talk in.
        - persona_username: The Discord username of the bot's persona.
                            This is case-sensitive and should match the dataset EXACTLY
                            for best results.
        - reply_probability: The probability, from 0 to 1, that the bot will respond
                             to any given message.
        - cold_call_probability: The probability, from 0 to 1, that the bot will 'cold call'
                                 every channel with a message initialised without priors.
        - cold_call_interval: The time interval between potential 'cold calls'.
        - history_buffer_size: The maximum number of messages in the history buffer.
        - mention_map: A dict mapping usernames to Discord user ids. Used to resolve mentions.
        - max_message_length: The maximum number of tokens for a single message.
    """
    channel_ids: Set[int]
    persona_username: str
    reply_probability: float
    cold_call_probability: float
    cold_call_interval: timedelta
    history_buffer_size: int
    mention_map: Dict[str, int]
    max_message_length: int
    # Private Instance Attributes:
    #   - _model: The GPT2 model.
    #   - _tokenizer: The GPT2 tokenizer.
    #   - _history_buffers: A dict mapping each channel id to its history buffer.
    #   - _previous_token_lengths: A dict mapping each channel id to the previous length of the
    #                              prompt, in terms of tokens.
    #   - _decode_regex: Regex pattern to decode message.
    _model: PreTrainedModel
    _tokenizer: PreTrainedTokenizer
    _history_buffers: Dict[int, Deque[discord.Message]]
    _previous_token_lengths: Dict[int, int]
    _decode_regex: Pattern

    def __init__(self, channel_ids: Set[int],
                 persona_username: str,
                 model_checkpoint: Union[str, Path],
                 reply_probability: float = 0.20,
                 cold_call_probability: float = 0.20,
                 cold_call_interval: timedelta = timedelta(hours=1),
                 history_buffer_size: int = 5,
                 no_cuda: bool = False,
                 use_fp16: bool = False,
                 mention_map: Optional[Dict[str, int]] = None,
                 max_message_length: int = 32) -> None:
        """Initialise the Client with the given server and channel ids,
        with a given GPT2 model checkpoint.
        """
        super().__init__()
        self.channel_ids = channel_ids
        self.persona_username = persona_username
        self.reply_probability = reply_probability
        self.cold_call_probability = cold_call_probability
        self.cold_call_interval = cold_call_interval
        self.history_buffer_size = history_buffer_size
        self.mention_map = mention_map if mention_map is not None else dict()
        self.max_message_length = max_message_length

        self._history_buffers = dict()
        self._previous_token_lengths = dict()
        self._model, self._tokenizer = load_model(
            model_checkpoint,
            no_cuda=no_cuda,
            quantize=use_fp16
        )
        # TODO: Add proper logging
        print(f'Loaded GPT2 model (checkpoint={model_checkpoint})')

        self._decode_regex = re.compile((
            f'^{re.escape(self._tokenizer.bos_token)}(?P<username>.+?)'
            f'(?:<sep>(?P<message>.+?))*'
            f'{re.escape(self._tokenizer.eos_token)}'
        ), flags=re.MULTILINE | re.DOTALL)

    async def on_ready(self) -> None:
        """Called when the client is ready."""
        print('Logged on as', self.user)

    def _make_prompt(self, message: discord.Message, separator_token: str = '<sep>') -> str:
        """Return a formatted prompt for the model given a message."""
        history = self._history_buffers.get(message.channel.id, list())
        history.append(message)

        formatted_history = ''.join(
            f'{self._tokenizer.bos_token}{x.author.name}{separator_token}{x.clean_content}' +
            f'{self._tokenizer.eos_token}' for x in history
        )
        return formatted_history + f'{self._tokenizer.bos_token}{self.persona_username}{separator_token}'

    def _resolve_mentions(self, message: str, server: discord.Guild) -> str:
        """Return a new string containing formatted mentions according to the mention map."""
        pattern = re.compile(r'@[A-Za-z0-9_-]*')
        matches = re.finditer(pattern, message)
        for match in matches:
            username = match.group(0)[1:]
            if username not in self.mention_map:
                continue
            user_id = self.mention_map[username]
            user = server.get_member(user_id)
            if user is None:
                print(f'Could not get user with name {username} (id={user_id})')
                continue

            start, end = match.span()
            message = message[:start] + user.mention + message[end:]
        return message

    async def on_message(self, message: discord.Message) -> None:
        """Called when the bot receives a message."""
        # Don't respond to ourselves!
        if message.author == self.user:
            return

        # Only talk in flagged channels
        if message.channel.id not in self.channel_ids:
            return

        print(f'{message.author.name} said {message.clean_content}')
        if random.random() > self.reply_probability:
            return

        async with message.channel.typing():
            channel_id = message.channel.id
            if channel_id not in self._previous_token_lengths:
                self._previous_token_lengths[channel_id] = 0

            prompt = self._make_prompt(message)

            max_gen_length = self._previous_token_lengths[channel_id] + self.max_message_length
            sample = generate(
                self._model,
                self._tokenizer,
                prompt,
                max_length=max_gen_length,
                # TODO: Vary temperature
                temperature=1.0
            )[0]

            self._previous_token_lengths[channel_id] = len(sample.tokens)

            tail_message = sample.raw_text[sample.raw_text.rfind(self._tokenizer.bos_token):]
            match = self._decode_regex.match(tail_message)
            if not match:
                return

            groups = {
                'username': match.group('username'),
                'message': match.group('message')
            }

            if any(value is None for value in groups.values()):
                return

            # Randomly delay to simulate typing
            await asyncio.sleep(random.randint(15, 60))

            output_message = _unescape_string(groups['message'])
            output_message = self._resolve_mentions(output_message, message.guild)
            await message.channel.send(output_message)

       # Append to history
        if channel_id not in self._history_buffers:
            self._history_buffers[channel_id] = collections.deque(maxlen=self.history_buffer_size)
        self._history_buffers[channel_id].append(message)


def main(args: argparse.Namespace) -> None:
    """Main entrypoint for the script."""
    if args.config_filepath is not None:
        with open(args.config_filepath, 'r') as config_file:
            config = json.load(config_file)
        # Update the args object
        for key, value in config.items():
            setattr(args, key, value)

    if args.channels is None or len(args.channels) == 0:
        raise ValueError('At least on channel id is required! Make sure you specified '
                         'the --channels flag in the command-line, or double check '
                         'your configuration file.')

    if args.persona_username is None:
        raise ValueError('A persona username is required! Make sure you specified '
                         'the --persona-username flag in the command-line, or double check '
                         'your configuration file.')

    if args.model_checkpoint is None:
        raise ValueError('A model checkpoint is required! Make sure you specified '
                         'the --model-checkpoint flag in the command-line, or double check '
                         'your configuration file.')

    if args.credentials_filepath is None:
        # Use the persona username to get a default credentials_filepath
        args.credentials_filepath = str(Path(f'instance/chat_bot/{args.persona_username}_credentials.json'))

    # Load credentials
    with open(args.credentials_filepath, 'r') as credentials_file:
        credentials = json.load(credentials_file)
        token = credentials.get('token', None)
        is_bot = credentials.get('bot', True)

    mention_map = None
    if args.mention_map_path is not None:
        with open(args.mention_map_path) as fp:
            mention_map = json.load(fp)

    # Set the event loop policy to selector on windows systems to avoid asyncio runtime errors.
    _set_selector_event_loop_policy()
    client = Client(args.channels,
                    args.persona_username,
                    args.model_checkpoint,
                    reply_probability=args.reply_probability,
                    cold_call_probability=args.cold_call_probability,
                    history_buffer_size=args.history_buffer_size,
                    mention_map=mention_map)
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(client.run(token, bot=is_bot))
    except:
        # Ignore exceptions!
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A discord chat bot that uses a GPT2 model '
                                                 'for message generation.')
    parser.add_argument('--credentials-filepath', type=Path, default=None,
                        help='The path to the JSON file containing the Discord client credentials.')
    parser.add_argument('--channels', nargs='+', help='The ids of the channels to talk in.',
                        type=int, default=None)
    parser.add_argument('--persona-username', type=str, help='The Discord username of the bot\'s persona.')
    parser.add_argument('--model-checkpoint', type=str, default=None,
                        help='The name of a standard pretrained model or a path to a checkpoint for '
                             'weights initialization.')
    parser.add_argument('--replay-probability', '-pr', type=float, default=0.2, dest='reply_probability',
                        help='The probability, from 0 to 1, that the bot will respond '
                             'to any given message.')
    parser.add_argument('--cold-call-probability', '-pcc', type=float, default=0.2, dest='cold_call_probability',
                        help='The probability, from 0 to 1, that the bot will \'cold call\' '
                             'every channel with a message initialised without priors.')
    parser.add_argument('--history-buffer-size', '-hs', type=int, default=5,
                        help='The maximum number of messages in the history buffer.')
    parser.add_argument('--mention-map', '-mm', type=Path, default=None, dest='mention_map_path',
                        help='The path to a JSON file containing a dict mapping names to user ids.')
    parser.add_argument('--config', '-c', dest='config_filepath', type=Path, default=None,
                        help='JSON file containing configuration arguments.'
                             'Can be used instead of supplying command-line arguments.')
    main(parser.parse_args())
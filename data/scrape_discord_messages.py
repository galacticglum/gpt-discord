"""Download messages from a Discord server and save it as a JSON file.
If no channel id is specified, messages are scraped from ALL channels."""
import json
import platform
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, AsyncGenerator

import asyncio
import discord
from tqdm import tqdm


def _set_selector_event_loop_policy() -> None:
    """Sets the asyncio event loop policy to SelectorEventLoop on Windows platform."""
    if platform.system() != 'Windows':
        return
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception as exception:
        print(f'Failed to set selector event loop policy! {exception}')


async def _history_iterator(history: discord.iterators.HistoryIterator) \
        -> AsyncGenerator[discord.Message, None]:
    """Iterate over a channel history, while silently handling exceptions."""
    while True:
        try:
            v = await history.next()
            yield v
        except discord.iterators.NoMoreItems:
            break
        except Exception as exception:
            print(exception)


class Client(discord.Client):
    """
    A discord.Client implementation for retrieving messages from a server.

    Instance Attributes:
        - server_id: The id of the server containing the target channel.
        - channel_ids: The target channel ids to scrape from.
                       If None, scrapes from all channels in the server.
        - limit: The maximum number of messages to get. If None, gets all.
        - output_filepath: The output JSON filepath. Defaults to "./messages.json".
    """
    def __init__(self, server_id: int, channel_ids: Optional[List[int]] = None,
                 limit: Optional[int] = None, output_filepath: Optional[Path] = None) -> None:
        """Initialise the Client with the given server and channel ids."""
        super().__init__()
        self.server_id = server_id
        self.channel_ids = channel_ids
        self.limit = limit
        self.output_filepath = output_filepath or Path('./messages.json')

    def _get_messages_from_channels(self, channels: List[discord.TextChannel]) \
            -> Dict[int, discord.iterators.HistoryIterator]:
        """Return a dict mapping the channel id to its history."""
        messages = {}
        for channel in channels:
            messages[channel.id] = channel.history(limit=self.limit, oldest_first=True)
        return messages

    async def on_ready(self) -> None:
        """Called when the client is ready."""
        guild = discord.utils.get(self.guilds, id=self.server_id)

        # Resolve channels
        if self.channel_ids is None:
            channels = guild.channels
        else:
            channels = [discord.utils.get(guild.channels, id=channel_id)
                        for channel_id in self.channel_ids]

        # Get only text channels
        channels = [channel for channel in channels if isinstance(channel, discord.TextChannel)]

        channels_str = ', '.join(f'#{channel.name}' for channel in channels)
        print(f'Scraping from {len(channels)} channels: {channels_str}')

        data = {}
        with tqdm(total=self.limit) as progress:
            message_histories = self._get_messages_from_channels(channels)
            for channel_id, history in message_histories.items():
                messages = []
                async for message in _history_iterator(history):
                    progress.update()
                    message = {
                        'id': message.id,
                        'author': {
                            'id': message.author.id,
                            'username': message.author.name,
                            'display_name': message.author.display_name,
                            'bot': message.author.bot
                        },
                        'content': message.content,
                        'clean_content': message.clean_content,
                        # Save the created_at attribute as a unix timestamp.
                        'created_at': datetime.timestamp(message.created_at)
                    }

                    messages.append(message)
                # Add to main object
                data[channel_id] = messages

        # Save data
        with open(self.output_filepath, 'w+') as output_file:
            json.dump(data, output_file)

        await self.close()


def main(args: argparse.Namespace) -> None:
    """Main entrypoint for the script."""
    # Load credentials
    with open(args.credentials_filepath, 'r') as credentials_file:
        credentials = json.load(credentials_file)
        token = credentials.get('token', None)
        is_bot = credentials.get('bot', True)

    if args.config_filepath is not None:
        with open(args.config_filepath, 'r') as config_file:
            config = json.load(config_file)
        # Update the args object
        for key, value in config.items():
            setattr(args, key, value)

    if args.server_id is None:
        raise ValueError('A server id is required! Make sure you specified the --server-id flag '
                         'in the command-line, or double check your configuration file.')

    # Set the event loop policy to selector on windows systems to avoid asyncio runtime errors.
    _set_selector_event_loop_policy()
    client = Client(args.server_id, args.channels,
                    limit=args.limit, output_filepath=args.output)
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(client.run(token, bot=is_bot))
    except:
        # Ignore exceptions!
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download messages from a Discord channel.')
    parser.add_argument('credentials_filepath', type=Path,
                        help='The path to the JSON file containing the Discord client credentials.')
    parser.add_argument('--server-id', '-sid', help='The server ID.', type=int, default=None)
    parser.add_argument('--channels', nargs='+', help='Channel ids to scrape from.',
                        type=int, default=None)
    parser.add_argument('--output', '-o', type=Path, default='messages.json',
                        help='The output filepath. Defaults to "./messages.json"')
    parser.add_argument('--limit', '-l', help='The number of chat messages to get. '
                        'If unspecified, gets all messages.', type=int, default=None)
    parser.add_argument('--config', '-c', dest='config_filepath', type=Path, default=None,
                        help='JSON file containing configuration arguments.'
                             'Can be used instead of supplying command-line arguments.')
    main(parser.parse_args())

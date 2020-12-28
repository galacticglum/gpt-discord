"""Download messages from a Discord channel and save it as a JSON file."""
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import discord
from tqdm import tqdm


class Client(discord.Client):
    """A discord.Client implementation for retrieving messages from a channel.

    Instance Attributes:
        - server_id: The id of the server containing the target channel.
        - channel_id: The target channel whose messages to retrieve.
        - limit: The maximum number of messages to get. If None, gets all.
        - output_filepath: The output JSON filepath. Defaults to "./messages.json".
    """

    def __init__(self, server_id: int, channel_id: int, limit: Optional[int] = None,
                 output_filepath: Optional[Path] = None) -> None:
        """Initialise the Client with the given server and channel ids."""
        super().__init__()
        self.server_id = server_id
        self.channel_id = channel_id
        self.limit = limit
        self.output_filepath = output_filepath or Path('./messages.json')

    async def on_ready(self) -> None:
        """Called when the client is ready."""
        guild = discord.utils.get(self.guilds, id=self.server_id)
        channel = discord.utils.get(guild.channels, id=self.channel_id)

        history = channel.history(limit=self.limit, oldest_first=True)
        # Serialize data
        messages = []
        with tqdm(total=self.limit) as progress:
            # Get each message, and add it to the buffer
            async for message in history:
                progress.update()

                data = {
                    'id': message.id,
                    'author': {
                        'id': message.author.id,
                        'username': message.author.name,
                        'display_name': message.author.display_name
                    },
                    'content': message.content,
                    'clean_content': message.clean_content,
                    # Save the created_at attribute as a unix timestamp.
                    'created_at': datetime.timestamp(message.created_at)
                }

                messages.append(data)

        # Save data
        with open(self.output_filepath, 'w+') as output_file:
            json.dump(messages, output_file)

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

    # Create the Discord client and run
    client = Client(args.server_id, args.channel_id,
                    limit=args.limit, output_filepath=args.output)
    client.run(token, bot=is_bot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download messages from a Discord channel.')
    parser.add_argument('credentials_filepath', type=Path,
                        help='The path to the JSON file containing the Discord client credentials.')
    parser.add_argument('--server-id', '-sid', help='The server ID.', type=int)
    parser.add_argument('--channel-id', '-cid', help='The channel ID.', type=int)
    parser.add_argument('--output', '-o', type=Path, default='messages.json',
                        help='The output filepath. Defaults to "./messages.json"')
    parser.add_argument('--limit', '-l', help='The number of chat messages to get. '
                        'If unspecified, gets all messages.', type=int, default=None)
    parser.add_argument('--config', '-c', dest='config_filepath', type=Path, default=None,
                        help='JSON file containing configuration arguments.'
                             'Can be used instead of supplying command-line arguments.')
    main(parser.parse_args())

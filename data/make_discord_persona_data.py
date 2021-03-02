"""Makes a persona-based dialogue dataset mapped by username given Discord messages.

This script creates a dataset where for each user (i.e. a persona), there is a list of candidate utterances
and dialogue history pairs.
"""
from __future__ import annotations

import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Union, Optional

from tqdm import tqdm
from discord.utils import escape_markdown


# A Regex pattern to match urls starting with or without http(s).
URL_MATCH_PATTERN = re.compile(
    r'(?i)(https?:\/\/(?:www\.|(?!www))'
    r'[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|'
    r'www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|'
    r'https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|'
    r'www\.[a-zA-Z0-9]+\.[^\s]{2,})'
)


@dataclass(frozen=True, eq=True)
class User:
    """A discord user.

    Instance Attributes:
        - id: The discord id of the user.
        - username: The name of the user.
        - display_name: The user's display name.
        - is_bot: Whether the user is a bot.
    """
    id: int
    username: str
    display_name: str
    is_bot: bool

    @staticmethod
    def from_dict(data: dict) -> User:
        """Return the User object represented by the given dict."""
        return User(
            id=data['id'],
            username=data['username'],
            display_name=data['display_name'],
            is_bot=data.get('bot', False)
        )


@dataclass(frozen=True, eq=True)
class Message:
    """A discord message.

    Instance Attributes:
        - id: The discord id of the message.
        - author: A User object representing the author of the message.
        - content: The content of the message.
        - clean_content: The content of the message with discord special tokens removed.
        - created_at: A datetime.datetime object representing when the message was created."""
    id: int
    author: User
    content: str
    clean_content: str
    created_at: datetime

    @staticmethod
    def from_dict(data: dict) -> Message:
        """Return the Message object represented by the given dict.

        This assumes that everything is encoded in a dict/json-like structure,
        including the user object. The "created_at" attribute should be given
        as a unix timestamp.
        """
        return Message(
            id=data['id'],
            author=User.from_dict(data['author']),
            content=data['content'],
            clean_content=data['clean_content'],
            created_at=datetime.fromtimestamp(data['created_at'])
        )


@dataclass
class PersonaUtterance:
    """A single utterance.

    Instance Attributes:
        - candidates: a list of candidate messages/utterances.
        - history: a list of messages, sorted in chronological order (old to new)
                   containing the chat history.
    """
    candidates: List[Message]
    history: List[Message]

    def to_json(self) -> dict:
        """Return this utterance as a json serializable object."""
        _encode_func = lambda x: escape_markdown(x.clean_content)
        return {
            'candidates': [_encode_func(x) for x in self.candidates],
            'history': [_encode_func(x) for x in self.history]
        }


@dataclass
class Persona:
    """A collection of utterances and message history for a single person/user.

    Instance Attributes:
        - user: The user corresponding to this persona.
        - utterances: A list of utterances.
    """
    user: User
    utterances: List[PersonaUtterance]

    def __init__(self, user: User, utterances: Optional[List[PersonaUtterance]] = None):
        """Initialise a Persona object."""
        self.user = user
        self.utterances = utterances or list()

    def to_json(self) -> dict:
        """Return this persona as a json serializable object."""
        return {
            'personality': [self.user.username],
            'utterances': [x.to_json() for x in self.utterances]
        }


def load_messages(file: Path, remove_links: bool = False) -> Union[List[Message], Dict[int, List[Message]]]:
    """Return the messages containined in the given json file.

    If the root level object is a list, then a list of messages is returned.
    Otherwise, if the root level object is dict, then the function returns a
    dict mapping channel ids to the messages in that channel.

    It is assumed that the data matches one of these two formats.
    """
    def _messages_to_obj(raw_messages: List[dict]) -> List[Message]:
        """Return a list of message objects."""
        result = []
        for raw_message in raw_messages:
            if remove_links:
                # Remove links from the content and clean_content attributes
                raw_message['content'] = re.sub(URL_MATCH_PATTERN, '', raw_message['content'])
                raw_message['clean_content'] = re.sub(URL_MATCH_PATTERN, '', raw_message['clean_content'])
            result.append(Message.from_dict(raw_message))
        return result

    with open(file) as fp:
        data = json.load(fp)

    if isinstance(data, list):
        return _messages_to_obj(data)
    elif isinstance(data, dict):
        return {
            channel: _messages_to_obj(messages)
            for channel, messages in data.items()
        }
    else:
        raise ValueError('The file contained an invalid data format. See function docstring.')


def get_user_message_count(messages: List[Message]) -> Dict[User, int]:
    """Return a dict mapping each discord user to the number of messages
    in the given list of messages.

    Preconditions:
        - every element in messages has an 'author' key with a dict value,
          that itself contains an 'id' key with an integer value specifying
          the discord user id of the author of the message.
    """
    user_message_count = {}
    for message in messages:
        if message.author not in user_message_count:
            user_message_count[message.author] = 0
        user_message_count[message.author] += 1
    return user_message_count


def main(args: argparse.Namespace) -> None:
    """Main entrypoint for the script."""
    personas = {}
    channels = load_messages(args.input_file, remove_links=args.remove_links)
    for channel_messages in channels.values():
        # Sort by message time to get them in the correct order!
        channel_messages.sort(key=lambda x: x.created_at)
        user_message_count = get_user_message_count(channel_messages)
        for user, message_count in user_message_count.items():
            # Skip this user if they don't match the message count threshold,
            # or if there already exists an entry for this user!
            if message_count < args.persona_message_threshold or user in personas:
                continue
            personas[user] = Persona(user)

        for i, message in enumerate(channel_messages):
            # Skip this message if we aren't making a persona for the user.
            if message.author not in personas:
                continue

            # Get messages for utterance
            history = channel_messages[max(0, i - args.max_history - 1):i]
            candidates = channel_messages[i:i + args.max_candidates + 1]
            utterance = PersonaUtterance(candidates, history)
            # Add the utterance to this user's persona
            persona = personas[message.author]
            persona.utterances.append(utterance)

    with open(args.output_file, 'w+') as output_file:
        serialized_data = []
        with tqdm(personas) as progress_bar:
            for user, persona in personas.items():
                progress_bar.set_description(f'Writing data for \'{user.username}\'')
                serialized_data.append(persona.to_json())
                progress_bar.update()
        print(f'Saving to \'{args.output_file}\'')
        # TODO: Split into training and evaluation sets!
        json.dump({'train': serialized_data, 'valid': []}, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Makes a persona-based dialogue dataset mapped by username '
                    'given Discord messages.'
    )
    parser.add_argument('input_file', type=Path,
                        help='JSON filepaths containing a JSON dict mapping each channel to a list '
                              'of Discord messages. A channel is defined a continuous stream of messages.')
    parser.add_argument('output_file', type=Path, help='File to output the corpus.')
    parser.add_argument('--max-history', type=int, default=32,
                        help='The maximum number of messages to look behind.')
    parser.add_argument('--max-candidates', type=int, default=16,
                        help='The maximum number of candidates messages to look ahead.')
    parser.add_argument('--ignore-bots', dest='ignore_bots', action='store_true',
                        help='Whether to ignore messages made by bots.')
    parser.add_argument('--remove-links', dest='remove_links', action='store_true',
                        help='Whether to remove links from messages.')
    parser.add_argument('--min-messages', dest='persona_message_threshold', type=int, default=512,
                        help='The minumum message threshold to create a persona for a user.')
    main(parser.parse_args())

"""Makes a corpus of Discord messages, mapped by username."""
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from discord.utils import escape_markdown


BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
SEPARATOR_TOKEN = '<sep>'


def main(args: argparse.Namespace) -> None:
    """Main entrypoint for the script."""
    messages = []
    for filepath in args.input_files:
        with open(filepath) as file:
            data = json.load(file)
            if isinstance(data, dict):
                for x in data.values():
                    messages.extend(x)
            elif isinstance(data, list):
                messages.extend(data)
            else:
                raise ValueError(f'Invalid format for input file \'{filepath}\'!')

    if args.sort_by_time:
        messages.sort(key=lambda x: x['created_at'])

    with open(args.output_file, 'wb+') as output_file:
        for message in tqdm(messages):
            if args.ignore_bots and message['author'].get('bot', False):
                # Ignore bot messages
                continue
            if not message.get('clean_content', None):
                # Skip empty messages
                continue

            sample = '{}{}{}{}{}'.format(
                args.start_token,
                message['author']['username'],
                args.separator_token,
                escape_markdown(message['clean_content']),
                args.end_token
            ).encode('unicode_escape') + b'\n'
            output_file.write(sample)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Makes a corpus of Discord messages, mapped by username.'
    )
    parser.add_argument('input_files', type=Path, nargs='+',
                        help='JSON filepaths containing a list of Discord messages, or '
                              'a JSON object mapping channel ids to a list of messages.')
    parser.add_argument('output_file', type=Path, help='File to output the corpus.')
    parser.add_argument('--start-token', type=str, default=BOS_TOKEN,
                        help='The start BOS (beginning of sentence) token. Defaults to \'<bos>\'.')
    parser.add_argument('--separator-token', type=str, default=SEPARATOR_TOKEN,
                        help='The separator token (between the name and messasge content). '
                             'Defaults to \'<sep>\'.')
    parser.add_argument('--end-token', type=str, default=EOS_TOKEN,
                        help='The end EOS (end of sentence) token. Defaults to \'<eos>\'.')
    parser.add_argument('--sort', dest='sort_by_time', action='store_true',
                        help='Whether to sort by post time.')
    parser.add_argument('--ignore-bots', dest='ignore_bots', action='store_true',
                        help='Whether to ignore messages made by bots.')
    main(parser.parse_args())

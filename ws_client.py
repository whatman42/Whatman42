#!/usr/bin/env python3
"""
Simple command line client for Testing/debugging
a Freqtrade bot's message websocket

Should not import anything from freqtrade,
so it can be used as a standalone script.
"""

import argparse
import asyncio
import logging
import socket
import sys
import time
from pathlib import Path

import orjson
import pandas
import rapidjson
import websockets


logger = logging.getLogger("WebSocketClient")


# ---------------------------------------------------------------------------


def setup_logging(filename: str):
    )


def parse_args():
    return vars(args)


def load_config(configfile):
        sys.exit(1)


def readable_timedelta(delta):
    return f"{int(minutes)}:{int(seconds)}.{int(milliseconds)}"


# ----------------------------------------------------------------------------


def json_serialize(message):
    return str(orjson.dumps(message), "utf-8")


def json_deserialize(message):
    return rapidjson.loads(message, object_hook=_json_object_hook)


# ---------------------------------------------------------------------------


class ClientProtocol:
        self.logger.info(data)


async def create_client(
            continue


# ---------------------------------------------------------------------------


async def _main(args):
    )


def main():
    args = parse_args()
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        logger.info("Exiting...")


if __name__ == "__main__":
    main()
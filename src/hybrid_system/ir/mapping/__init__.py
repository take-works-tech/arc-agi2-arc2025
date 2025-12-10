#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
コマンド→テンプレート変換の公開インターフェース
"""

from .command_registry import (
    CommandTemplate,
    CommandArgumentSpec,
    get_command_template,
    resolve_command_template,
    list_supported_commands,
)
from .argument_resolver import resolve_argument_slots

__all__ = [
    "CommandTemplate",
    "CommandArgumentSpec",
    "get_command_template",
    "resolve_command_template",
    "list_supported_commands",
    "resolve_argument_slots",
]

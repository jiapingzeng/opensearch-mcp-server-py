# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Secrets are masked in the `Provided:` echo of a missing-field validation error."""

import copy
import pytest
from mcp_server_opensearch import global_state
from tools.tool_params import (
    GetIndexMappingArgs,
    _is_secret_key,
    _redact_secrets,
    baseToolArgs,
    validate_args_for_mode,
)


@pytest.fixture
def mode():
    """Set the server mode for one test and restore the previous one afterwards."""
    previous = global_state._current_mode

    def _set(value):
        global_state.set_mode(value)

    yield _set
    global_state._current_mode = previous


def _provided(args, model=GetIndexMappingArgs):
    with pytest.raises(ValueError) as excinfo:
        validate_args_for_mode(args, model)
    return str(excinfo.value), excinfo.value


# Password is masked in single mode.
def test_password_masked_in_single_mode(mode):
    mode('single')
    message, _ = _provided({'opensearch_username': 'admin', 'opensearch_password': 'hunter2'})

    assert message == (
        "Missing required field: 'index'\n\n"
        "Provided: {'opensearch_username': 'admin', 'opensearch_password': '***'}"
    )
    assert 'hunter2' not in message


# Password is masked in multi mode.
def test_password_masked_in_multi_mode(mode):
    mode('multi')
    message, _ = _provided({'opensearch_url': 'http://x:9200', 'opensearch_password': 'hunter2'})

    assert message == (
        "Missing required fields: 'opensearch_cluster_name', 'index'\n\n"
        "Provided: {'opensearch_url': 'http://x:9200', 'opensearch_password': '***'}"
    )
    assert 'hunter2' not in message


# Single mode still drops opensearch_cluster_name; exception type and chaining unchanged.
def test_single_mode_keeps_cluster_name_removal_and_chaining(mode):
    mode('single')
    message, exc = _provided({'opensearch_cluster_name': 'c1', 'opensearch_password': 'p'})

    assert message == "Missing required field: 'index'\n\nProvided: {'opensearch_password': '***'}"
    assert type(exc) is ValueError
    assert exc.__cause__ is not None


# aws_session_token is masked.
def test_aws_session_token_masked(mode):
    mode('single')
    message, _ = _provided({'aws_session_token': 'FwoGZXIvYXdzE'})

    assert "'aws_session_token': '***'" in message
    assert 'FwoGZXIvYXdzE' not in message


# An Authorization key inside a nested dict is masked.
def test_nested_authorization_masked(mode):
    mode('single')
    args = {'headers': {'Authorization': 'Basic YWRtaW46aHVudGVyMg==', 'Accept': 'json'}}
    message, _ = _provided(args)

    assert "Provided: {'headers': {'Authorization': '***', 'Accept': 'json'}}" in message
    assert 'YWRtaW46aHVudGVyMg' not in message


# A secret key inside a list of dicts is masked.
def test_key_inside_list_of_dicts_masked(mode):
    mode('multi')
    args = {'items': [{'name': 'a', 'api_key': 'k1'}, {'name': 'b', 'client_secret': 's2'}, 3]}
    message, _ = _provided(args)

    assert (
        "Provided: {'items': [{'name': 'a', 'api_key': '***'}, "
        "{'name': 'b', 'client_secret': '***'}, 3]}"
    ) in message
    assert 'k1' not in message
    assert 's2' not in message


# Non-secret fields (index, opensearch_url, opensearch_username, tokenizer) are unchanged.
def test_non_secret_fields_unchanged():
    args = {
        'index': 'logs',
        'opensearch_url': 'https://search.example:9200',
        'opensearch_username': 'admin',
        'tokenizer': 'standard',
    }

    assert _redact_secrets(args) == args


# Non-secret fields stay visible in the actual error message too.
def test_non_secret_fields_visible_in_message(mode):
    mode('single')
    message, _ = _provided({'opensearch_username': 'admin', 'tokenizer': 'standard'})

    assert message.endswith("Provided: {'opensearch_username': 'admin', 'tokenizer': 'standard'}")


# The rule: listed substrings, exact `token` or a `_token` suffix; no bare `token` substring.
@pytest.mark.parametrize(
    'key',
    [
        'password',
        'OPENSEARCH_PASSWORD',
        'db_passwd',
        'client_secret',
        'aws_credentials',
        'Authorization',
        'Cookie',
        'set_cookie',
        'api_key',
        'ApiKey',
        'aws_access_key_id',
        'private_key',
        'token',
        'Token',
        'aws_session_token',
        'refresh_token',
    ],
)
def test_secret_keys_match(key):
    assert _is_secret_key(key)


@pytest.mark.parametrize(
    'key',
    ['tokenizer', 'tokens', 'token_count', 'index', 'opensearch_username', 'aws_region', 1, None],
)
def test_non_secret_keys_do_not_match(key):
    assert not _is_secret_key(key)


# Every baseToolArgs field that holds a credential is caught by the rule.
def test_base_tool_args_credential_fields_caught():
    caught = {name for name in baseToolArgs.model_fields if _is_secret_key(name)}

    assert caught == {'opensearch_password'}


# The original args_dict is not mutated.
def test_original_args_not_mutated(mode):
    args = {
        'opensearch_cluster_name': 'c1',
        'opensearch_password': 'hunter2',
        'headers': {'Authorization': 'Bearer abc'},
        'items': [{'api_key': 'k1'}],
    }
    snapshot = copy.deepcopy(args)

    for value in ('single', 'multi'):
        mode(value)
        _provided(args)
        assert args == snapshot

    assert _redact_secrets(args) is not args
    assert args == snapshot

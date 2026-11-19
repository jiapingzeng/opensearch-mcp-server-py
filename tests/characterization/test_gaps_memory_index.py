# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Characterization gaps: the memory index auto-create on the first ``SaveMemoryTool`` call.

``save_memory_tool`` first calls ``_ensure_memory_index`` (``src/tools/memory_tools.py``):
when ``HEAD /<memory index>`` answers 404 the index is created through boto3, not through
OpenSearch, and only then is the memory indexed.

- On OpenSearch Serverless (``AWS_OPENSEARCH_SERVERLESS=true``) the tool first makes sure a
  data access policy named ``mcp-memory-access`` covers the caller
  (``_ensure_aoss_data_access_policy``): STS ``GetCallerIdentity`` (an assumed-role ARN is
  turned into its IAM role ARN), ``BatchGetCollection`` for the collection name,
  ``GetAccessPolicy``, then ``UpdateAccessPolicy`` or ``CreateAccessPolicy`` as needed; every
  failure after the collection lookup is logged and swallowed. Then
  ``opensearchserverless`` ``CreateIndex``, where a conflict or an "already exists" error
  counts as success.
- On a managed domain (``AWS_OPENSEARCH_DOMAIN_NAME``) the tool calls ``opensearch``
  ``CreateIndex`` with camelCase parameter names the API does not have, so botocore's
  parameter validation rejects the call before it is signed or sent, whatever the service
  would answer: the success log and both "already exists" branches after it are unreachable
  today (``test_managed_domain_create_index_is_rejected_before_it_is_sent``).

Each case launches the server with ``memory_enabled_session`` (``MEMORY_TOOLS_ENABLED=true``),
``OPENSEARCH_NO_AUTH`` so the memory documents reach the fake OpenSearch, and the fake
static keys of ``boto3_stub.env``; ``boto3_stub`` answers every boto3 call (``harness``
README, ``boto3_stub``). The golden, under ``golden/gaps_memory_index/``, holds the canned
boto3 replies, the call and its result, the boto3 calls the server made, the requests the
fake OpenSearch saw for the call and the log records of ``tools.memory_tools`` and
``tools.tool_logging``. The server runs with ``--debug --log-format json``: the branches
that count as success differ only in a DEBUG record, and JSON records are parsed by logger
without their timestamps.

The replies follow the stub's limits (13c review): only the operations above, of the
``query`` (``sts``), ``json`` (``opensearchserverless``) and ``rest-json`` (``opensearch``)
services; every success reply has status 200; every output member given is a body member
(none is read from a header or the status code, which the stub would drop silently); no
output is streamed.

Canned data uses fixed dates far from today (``createdDate`` 2020-01-01, see README.md).
"""

import json
import pytest
from dataclasses import asdict, dataclass
from typing import Any


AREA = 'gaps_memory_index'
OPENSEARCH_VERSION = '3.3.0'
TOOL = 'SaveMemoryTool'

REGION = 'eu-west-1'
ACCOUNT = '123456789012'
SESSION_ARN = f'arn:aws:sts::{ACCOUNT}:assumed-role/MemoryWriter/char-session'
ROLE_ARN = f'arn:aws:iam::{ACCOUNT}:role/MemoryWriter'
USER_ARN = f'arn:aws:iam::{ACCOUNT}:user/memory-writer'
OTHER_ROLE_ARN = f'arn:aws:iam::{ACCOUNT}:role/SearchReader'
OTHER_USER_ARN = f'arn:aws:iam::{ACCOUNT}:user/ops'
COLLECTION_ID = 'k2x9memcoll'
COLLECTION_NAME = 'agent-memories'
DOMAIN = 'agent-domain'
MEMORY_INDEX = 'agent-memory'
POLICY_NAME = 'mcp-memory-access'
POLICY_VERSION = 'MTU3NzgzNjgwMDAwMF8x'
# 2020-01-01T00:00:00Z in epoch milliseconds, far from today (README.md).
CREATED_MS = 1577836800000

NO_AUTH = {'OPENSEARCH_NO_AUTH': 'true'}
SERVERLESS_ENV = {
    **NO_AUTH,
    'AWS_OPENSEARCH_SERVERLESS': 'true',
    'AWS_OPENSEARCH_COLLECTION_ID': COLLECTION_ID,
    'AWS_REGION': REGION,
}
DOMAIN_ENV = {**NO_AUTH, 'AWS_OPENSEARCH_DOMAIN_NAME': DOMAIN, 'AWS_REGION': REGION}
# DEBUG, so the branches that count as success are told apart; JSON, so records parse.
SERVER_ARGS = ('--debug', '--log-format', 'json')
# The loggers whose records the goldens keep: the auto-create's and the tool error event's.
LOGGERS = ('tools.memory_tools', 'tools.tool_logging')

SAVE_ARGS = {'memory': 'User prefers vegetarian food', 'user_id': 'user-123'}
INDEXED = {
    '_index': MEMORY_INDEX,
    '_id': 'mem-7Kx2',
    '_version': 1,
    'result': 'created',
    '_shards': {'total': 2, 'successful': 1, 'failed': 0},
    '_seq_no': 0,
    '_primary_term': 1,
}

# PYTHONHASHSEED values one case also runs under (at least two, per the task).
HASH_SEEDS = ('0', '1')


# ---------------------------------------------------------------------------
# canned boto3 replies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ok:
    """A success reply (status 200) with output ``data``."""

    service: str
    operation: str
    data: Any = None

    def give(self, stub) -> None:
        stub.reply(self.service, self.operation, self.data)


@dataclass(frozen=True)
class Fail:
    """An AWS error reply: ``code``, ``message``, the modeled status (else 400)."""

    service: str
    operation: str
    code: str
    message: str

    def give(self, stub) -> None:
        stub.error(self.service, self.operation, self.code, self.message)


AOSS = 'opensearchserverless'


def identity(arn: str) -> Ok:
    return Ok(
        'sts', 'GetCallerIdentity', {'UserId': 'AIDAFAKEID:char', 'Account': ACCOUNT, 'Arn': arn}
    )


def access_policy(rule_sets: list[dict]) -> dict:
    return {
        'type': 'data',
        'name': POLICY_NAME,
        'policyVersion': POLICY_VERSION,
        'policy': rule_sets,
        'createdDate': CREATED_MS,
        'lastModifiedDate': CREATED_MS,
    }


def rule_set(collection: str, principals: list[str] | None) -> dict:
    rules = {
        'Rules': [
            {
                'Resource': [f'index/{collection}/*'],
                'Permission': ['aoss:ReadDocument'],
                'ResourceType': 'index',
            }
        ],
    }
    return rules if principals is None else {**rules, 'Principal': principals}


ASSUMED_ROLE = identity(SESSION_ARN)
COLLECTION = Ok(
    AOSS,
    'BatchGetCollection',
    {
        'collectionDetails': [
            {
                'id': COLLECTION_ID,
                'name': COLLECTION_NAME,
                'status': 'ACTIVE',
                'createdDate': CREATED_MS,
            }
        ],
        'collectionErrorDetails': [],
    },
)
NO_POLICY = Fail(AOSS, 'GetAccessPolicy', 'ResourceNotFoundException', 'Policy not found')
POLICY_CREATED = Ok(
    AOSS,
    'CreateAccessPolicy',
    {'accessPolicyDetail': access_policy([rule_set(COLLECTION_NAME, [ROLE_ARN])])},
)
INDEX_CREATED = Ok(AOSS, 'CreateIndex')
# The policy is created for the caller, then the index: the first save on a new collection.
FIRST_SAVE = (ASSUMED_ROLE, COLLECTION, NO_POLICY, POLICY_CREATED, INDEX_CREATED)


def replacing(replies: tuple, *changes) -> tuple:
    """``replies`` with each ``change`` in place of the reply to its operation, or added."""
    by_operation = {(r.service, r.operation): r for r in replies}
    for change in changes:
        by_operation[(change.service, change.operation)] = change
    return tuple(by_operation.values())


def existing_policy(principals: list[list[str] | None]) -> Ok:
    """``GetAccessPolicy`` answering the policy read: one rule set per entry of ``principals``."""
    sets = [rule_set(f'{COLLECTION_NAME}-{i}', p) for i, p in enumerate(principals)]
    return Ok(AOSS, 'GetAccessPolicy', {'accessPolicyDetail': access_policy(sets)})


POLICY_UPDATED = Ok(
    AOSS,
    'UpdateAccessPolicy',
    {'accessPolicyDetail': access_policy([rule_set(COLLECTION_NAME, [OTHER_ROLE_ARN, ROLE_ARN])])},
)


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One first save on a serverless collection: the boto3 replies and the calls expected."""

    name: str
    replies: tuple
    # The boto3 operations the save makes, in order (also in the golden, in full).
    operations: tuple
    # Whether the memory document is indexed after the auto-create (POST /<index>/_doc).
    indexed: bool = True


POLICY_OPS = ('GetCallerIdentity', 'BatchGetCollection', 'GetAccessPolicy')
FIRST_SAVE_OPS = (*POLICY_OPS, 'CreateAccessPolicy', 'CreateIndex')

SERVERLESS_CASES = (
    # --- the data access policy (_ensure_aoss_data_access_policy) ---
    # No policy yet: one is created for the caller's IAM role (the assumed-role session ARN
    # turned into the role ARN), scoped to the collection and the memory index, and the
    # index is created.
    Case('policy_created_then_index_created', FIRST_SAVE, FIRST_SAVE_OPS),
    # The collection id names no collection: the save fails before any policy call.
    Case(
        'collection_not_found',
        replacing(
            FIRST_SAVE,
            Ok(
                AOSS, 'BatchGetCollection', {'collectionDetails': [], 'collectionErrorDetails': []}
            ),
        ),
        ('GetCallerIdentity', 'BatchGetCollection'),
        indexed=False,
    ),
    # The collection lookup itself fails: not caught, so the save fails.
    Case(
        'collection_lookup_error',
        replacing(
            FIRST_SAVE,
            Fail(AOSS, 'BatchGetCollection', 'ValidationException', 'Invalid collection id'),
        ),
        ('GetCallerIdentity', 'BatchGetCollection'),
        indexed=False,
    ),
    # The policy already lists the caller (an IAM user ARN, kept as it is): no policy is
    # written, the index is created.
    Case(
        'policy_already_covers_caller',
        replacing(
            FIRST_SAVE,
            identity(USER_ARN),
            existing_policy([[OTHER_ROLE_ARN], [OTHER_USER_ARN, USER_ARN]]),
        ),
        (*POLICY_OPS, 'CreateIndex'),
    ),
    # The policy does not list the caller: the caller is added to every rule set and the
    # policy updated at the version read, then the index is created.
    Case(
        'policy_updated_for_caller',
        replacing(
            FIRST_SAVE,
            existing_policy([[OTHER_ROLE_ARN], [OTHER_USER_ARN, OTHER_ROLE_ARN]]),
            POLICY_UPDATED,
        ),
        (*POLICY_OPS, 'UpdateAccessPolicy', 'CreateIndex'),
    ),
    # characterization: current behaviour. A rule set without a `Principal` list makes
    # `rule_set['Principal'].append` raise KeyError, which the policy check catches as any
    # error: a warning naming only `'Principal'`, no update, and the index is created anyway.
    Case(
        'policy_rule_set_without_principal',
        replacing(FIRST_SAVE, existing_policy([[OTHER_ROLE_ARN], None]), POLICY_UPDATED),
        (*POLICY_OPS, 'CreateIndex'),
    ),
    # The update fails (here a version conflict): a warning, and the index is created.
    Case(
        'policy_update_fails',
        replacing(
            FIRST_SAVE,
            existing_policy([[OTHER_ROLE_ARN]]),
            Fail(AOSS, 'UpdateAccessPolicy', 'ConflictException', 'Policy version is stale'),
        ),
        (*POLICY_OPS, 'UpdateAccessPolicy', 'CreateIndex'),
    ),
    # characterization: current behaviour. The update answers ResourceNotFoundException
    # (the policy went away after it was read): the handler meant for the lookup catches it,
    # so a new policy is created for the caller alone, without the principals read before.
    Case(
        'policy_update_not_found_creates_policy',
        replacing(
            FIRST_SAVE,
            existing_policy([[OTHER_ROLE_ARN]]),
            Fail(AOSS, 'UpdateAccessPolicy', 'ResourceNotFoundException', 'Policy not found'),
        ),
        (*POLICY_OPS, 'UpdateAccessPolicy', 'CreateAccessPolicy', 'CreateIndex'),
    ),
    # Another error whose text names ResourceNotFoundException counts as no policy.
    Case(
        'policy_lookup_not_found_by_message',
        replacing(
            FIRST_SAVE,
            Fail(
                AOSS,
                'GetAccessPolicy',
                'ValidationException',
                f'ResourceNotFoundException: no data policy named {POLICY_NAME}',
            ),
        ),
        FIRST_SAVE_OPS,
    ),
    # Any other lookup error: a warning, no policy is written, and the index is created.
    Case(
        'policy_lookup_fails',
        replacing(
            FIRST_SAVE,
            Fail(AOSS, 'GetAccessPolicy', 'AccessDeniedException', 'Not authorized to read'),
        ),
        (*POLICY_OPS, 'CreateIndex'),
    ),
    # The policy appeared in between (a race): DEBUG only, and the index is created.
    Case(
        'policy_create_conflict',
        replacing(
            FIRST_SAVE,
            Fail(AOSS, 'CreateAccessPolicy', 'ConflictException', 'Policy already exists'),
        ),
        FIRST_SAVE_OPS,
    ),
    # Any other create error: a warning, and the index is created.
    Case(
        'policy_create_fails',
        replacing(
            FIRST_SAVE,
            Fail(AOSS, 'CreateAccessPolicy', 'ServiceQuotaExceededException', 'Quota reached'),
        ),
        FIRST_SAVE_OPS,
    ),
    # --- create_index ---
    # (Success: policy_created_then_index_created.)
    # A conflict (modeled, ConflictException): the index exists, the memory is saved.
    Case(
        'index_conflict',
        replacing(FIRST_SAVE, Fail(AOSS, 'CreateIndex', 'ConflictException', 'Index exists')),
        FIRST_SAVE_OPS,
    ),
    # Another error whose text says "already exists", in any case: the same.
    Case(
        'index_already_exists_message',
        replacing(
            FIRST_SAVE,
            Fail(
                AOSS, 'CreateIndex', 'ValidationException', f'Index {MEMORY_INDEX} Already Exists'
            ),
        ),
        FIRST_SAVE_OPS,
    ),
    # An error coded ResourceAlreadyExistsException (not modeled for this API, so a plain
    # ClientError), whose message does not say "already exists": the same.
    Case(
        'index_already_exists_code',
        replacing(
            FIRST_SAVE,
            Fail(AOSS, 'CreateIndex', 'ResourceAlreadyExistsException', 'Index is present'),
        ),
        FIRST_SAVE_OPS,
    ),
    # Any other error: raised, the save fails and nothing is indexed.
    Case(
        'index_create_fails',
        replacing(
            FIRST_SAVE,
            Fail(AOSS, 'CreateIndex', 'ValidationException', 'Invalid index schema'),
        ),
        FIRST_SAVE_OPS,
        indexed=False,
    ),
)
SERVERLESS_IDS = [case.name for case in SERVERLESS_CASES]

# What the stub would answer the managed domain's create_index with, were it sent.
DOMAIN_REPLIES = (
    pytest.param(Ok('opensearch', 'CreateIndex', {'Status': 'CREATED'}), id='created'),
    pytest.param(
        Fail('opensearch', 'CreateIndex', 'ResourceAlreadyExistsException', 'Index exists'),
        id='already_exists',
    ),
    pytest.param(
        Fail(
            'opensearch', 'CreateIndex', 'ValidationException', 'Index agent-memory already exists'
        ),
        id='already_exists_message',
    ),
)


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def opensearch_version(fake_opensearch):
    fake_opensearch.version = OPENSEARCH_VERSION


def first_save_routes(fake_opensearch) -> None:
    """The memory index is missing (so the save creates it); the document is then indexed."""
    fake_opensearch.reply('HEAD', f'/{MEMORY_INDEX}', None, status=404)
    fake_opensearch.reply('POST', f'/{MEMORY_INDEX}/_doc', INDEXED, status=201)


def log_records(stderr: str) -> list[dict]:
    """The JSON log records of ``LOGGERS`` on the server's stderr, without their timestamps."""
    records = []
    for line in stderr.splitlines():
        if not line.startswith('{'):
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and record.get('logger') in LOGGERS:
            records.append({k: v for k, v in record.items() if k != 'timestamp'})
    return records


async def first_save(memory_enabled_session, boto3_stub, replies, env: dict) -> dict:
    """Launch the server with ``replies`` given, call the tool once, return what it did."""
    for reply in replies:
        reply.give(boto3_stub)
    async with memory_enabled_session(env={**env, **boto3_stub.env}, args=SERVER_ARGS) as session:
        await session.initialize()
        requests_before = len(session.opensearch_requests())
        calls_before = len(session.boto3_calls())
        result = await session.call_tool(TOOL, SAVE_ARGS)
        requests = session.opensearch_requests()[requests_before:]
        calls = session.boto3_calls()[calls_before:]
    # Read once the server has exited, so stderr is complete.
    logs = log_records(session.stderr)
    assert boto3_stub.unmatched == []
    return {
        'boto3_replies': [{'kind': type(r).__name__, **asdict(r)} for r in replies],
        'call': {'name': TOOL, 'arguments': SAVE_ARGS},
        'result': result,
        'boto3_calls': calls,
        'requests': requests,
        'logs': logs,
    }


def request_lines(recorded: dict) -> list[str]:
    return [f'{r["method"]} {r["path"]}' for r in recorded['requests']]


# ---------------------------------------------------------------------------
# serverless collection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('case', SERVERLESS_CASES, ids=SERVERLESS_IDS)
async def test_serverless_first_save(
    case, memory_enabled_session, boto3_stub, fake_opensearch, no_network, golden, normalize
):
    first_save_routes(fake_opensearch)

    recorded = await first_save(memory_enabled_session, boto3_stub, case.replies, SERVERLESS_ENV)

    assert tuple(c['operation'] for c in recorded['boto3_calls']) == case.operations
    expected_requests = [f'HEAD /{MEMORY_INDEX}']
    if case.indexed:
        expected_requests.append(f'POST /{MEMORY_INDEX}/_doc')
    assert request_lines(recorded) == expected_requests
    assert recorded['result']['isError'] is not case.indexed
    assert no_network == []
    golden(f'{AREA}/serverless_{case.name}', normalize(recorded))


@pytest.mark.parametrize('seed', HASH_SEEDS)
async def test_serverless_policy_update_under_hash_seed(
    seed, memory_enabled_session, boto3_stub, fake_opensearch, golden, normalize
):
    # The policy check collects the principals in a set; the update writes the rule sets
    # from the policy read, in order, so the golden is the same under any hash seed.
    case = next(c for c in SERVERLESS_CASES if c.name == 'policy_updated_for_caller')
    first_save_routes(fake_opensearch)

    recorded = await first_save(
        memory_enabled_session,
        boto3_stub,
        case.replies,
        {**SERVERLESS_ENV, 'PYTHONHASHSEED': seed},
    )

    golden(f'{AREA}/serverless_{case.name}', normalize(recorded))


# ---------------------------------------------------------------------------
# managed domain
# ---------------------------------------------------------------------------


# On a managed domain the tool calls `opensearch` create_index with DomainName, IndexName
# and IndexSchema (the schema as a document): botocore validates, signs and sends it, and
# the stub answers. A success is logged at INFO; ResourceAlreadyExistsException, or any
# other error whose text says "already exists", counts as success with a DEBUG record.
# With each of these replies the memory is then indexed. One golden per reply.
DOMAIN_OUTCOMES = {
    'created': ('INFO', f'Created memory index "{MEMORY_INDEX}" on domain {DOMAIN}'),
    'already_exists': ('DEBUG', f'Memory index "{MEMORY_INDEX}" already exists'),
    'already_exists_message': ('DEBUG', f'Memory index "{MEMORY_INDEX}" already exists'),
}


@pytest.mark.parametrize('reply', DOMAIN_REPLIES)
async def test_managed_domain_create_index(
    reply,
    request,
    memory_enabled_session,
    boto3_stub,
    fake_opensearch,
    no_network,
    golden,
    normalize,
):
    reply_id = request.node.callspec.id
    first_save_routes(fake_opensearch)

    recorded = await first_save(memory_enabled_session, boto3_stub, (reply,), DOMAIN_ENV)

    assert boto3_stub.calls == recorded['boto3_calls']
    assert [(c['service'], c['operation']) for c in recorded['boto3_calls']] == [
        ('opensearch', 'CreateIndex')
    ]
    params = recorded['boto3_calls'][0]['params']
    assert sorted(params) == ['DomainName', 'IndexName', 'IndexSchema']
    assert (params['DomainName'], params['IndexName']) == (DOMAIN, MEMORY_INDEX)
    assert isinstance(params['IndexSchema'], dict)
    assert request_lines(recorded) == [f'HEAD /{MEMORY_INDEX}', f'POST /{MEMORY_INDEX}/_doc']
    assert recorded['result']['isError'] is False
    level, message = DOMAIN_OUTCOMES[reply_id]
    assert {'level': level, 'logger': 'tools.memory_tools', 'message': message} in recorded['logs']
    assert no_network == []
    golden(f'{AREA}/managed_domain_create_index_{reply_id}', normalize(recorded))

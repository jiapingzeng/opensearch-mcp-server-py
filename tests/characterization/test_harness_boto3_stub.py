# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for ``boto3_stub`` (see conftest.py and README.md).

The stub answers boto3 calls from canned replies at botocore's ``before-send``, in the test
process and in a server ``stdio_launcher`` (``memory_enabled_session``) starts, for at least
the calls the memory index auto-create makes: ``sts`` ``GetCallerIdentity``,
``opensearchserverless`` ``BatchGetCollection``, ``GetAccessPolicy``, ``CreateAccessPolicy``
and ``CreateIndex``, and ``opensearch`` ``CreateIndex``. Each test also shows that nothing
reaches AWS: botocore's HTTP sender is never called and no socket leaves the machine.
"""

import boto3
import botocore.endpoint
import botocore.httpsession
import botocore.session
import json
import pytest
from botocore.exceptions import ClientError, NoCredentialsError, ParamValidationError


AREA = 'harness_boto3_stub'
REGION = 'eu-west-1'
ACCOUNT = '123456789012'
SESSION_ARN = f'arn:aws:sts::{ACCOUNT}:assumed-role/MemoryWriter/char-session'
ROLE_ARN = f'arn:aws:iam::{ACCOUNT}:role/MemoryWriter'
COLLECTION_ID = 'k2x9memcoll'
COLLECTION_NAME = 'agent-memories'
DOMAIN = 'agent-domain'
MEMORY_INDEX = 'agent-memory'
POLICY_NAME = 'mcp-memory-access'
SCHEMA = {'mappings': {'properties': {'memory_text': {'type': 'text'}}}}
# Canned data uses fixed dates far from today (README.md): 2020-01-01 in epoch milliseconds.
CREATED_MS = 1577836800000

POLICY_DETAIL = {
    'type': 'data',
    'name': POLICY_NAME,
    'policyVersion': 'MTU3NzgzNjgwMDAwMF8x',
    'policy': [{'Rules': [], 'Principal': [ROLE_ARN]}],
}

# (service, operation, method, arguments, reply data) of every operation the stub must answer.
SUPPORTED = [
    (
        'sts',
        'GetCallerIdentity',
        'get_caller_identity',
        {},
        {'UserId': 'AROAFAKEROLEID:char-session', 'Account': ACCOUNT, 'Arn': SESSION_ARN},
    ),
    (
        'opensearchserverless',
        'BatchGetCollection',
        'batch_get_collection',
        {'ids': [COLLECTION_ID]},
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
    ),
    (
        'opensearchserverless',
        'GetAccessPolicy',
        'get_access_policy',
        {'type': 'data', 'name': POLICY_NAME},
        {'accessPolicyDetail': POLICY_DETAIL},
    ),
    (
        'opensearchserverless',
        'CreateAccessPolicy',
        'create_access_policy',
        {'type': 'data', 'name': POLICY_NAME, 'policy': json.dumps(POLICY_DETAIL['policy'])},
        {'accessPolicyDetail': POLICY_DETAIL},
    ),
    (
        'opensearchserverless',
        'CreateIndex',
        'create_index',
        {'id': COLLECTION_ID, 'indexName': MEMORY_INDEX, 'indexSchema': SCHEMA},
        None,
    ),
    (
        'opensearch',
        'CreateIndex',
        'create_index',
        {'DomainName': DOMAIN, 'IndexName': MEMORY_INDEX, 'IndexSchema': SCHEMA},
        {'Status': 'CREATED'},
    ),
]


@pytest.fixture
def credentials(boto3_stub, monkeypatch):
    """The stub's fake static credentials, set in this process as a test's ``env`` sets them."""
    for name, value in boto3_stub.env.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def http_sends(monkeypatch) -> list[str]:
    """Every request that reaches botocore's HTTP layer (none should): recorded, then refused."""
    sent: list[str] = []

    def refuse(what: str):
        def send(self, request, *args, **kwargs):
            sent.append(f'{what}: {request.url}')
            raise AssertionError(f'a boto3 request reached {what}: {request.url}')

        return send

    monkeypatch.setattr(botocore.endpoint.Endpoint, '_send', refuse('Endpoint._send'))
    monkeypatch.setattr(botocore.httpsession.URLLib3Session, 'send', refuse('URLLib3Session.send'))
    return sent


def client(service: str):
    return boto3.Session(region_name=REGION).client(service)


def without_request_metadata(response: dict) -> dict:
    return {k: v for k, v in response.items() if k != 'ResponseMetadata'}


# --- in-process ------------------------------------------------------------------


def test_answers_each_supported_operation(
    boto3_stub, credentials, http_sends, no_network, golden, normalize
):
    for service, operation, _, _, data in SUPPORTED:
        boto3_stub.reply(service, operation, data)

    exchanges = []
    for service, operation, method, arguments, _ in SUPPORTED:
        response = getattr(client(service), method)(**arguments)
        exchanges.append({'call': [service, operation, arguments], 'response': response})

    for (_, _, _, _, data), exchange in zip(SUPPORTED, exchanges):
        assert without_request_metadata(exchange['response']) == (data or {})
    assert [(c['service'], c['operation']) for c in boto3_stub.calls] == [
        (service, operation) for service, operation, *_ in SUPPORTED
    ]
    # Every request was signed (SigV4, for the client's region and the service's signing name).
    assert all(c['auth']['kind'] == 'sigv4' for c in boto3_stub.calls)
    assert (http_sends, no_network) == ([], [])
    golden(
        f'{AREA}/in_process_replies',
        normalize({'exchanges': exchanges, 'calls': boto3_stub.calls}),
    )


ERRORS = [
    # (service, operation, method, arguments, code, modeled): json, rest-json and query.
    pytest.param(
        'opensearchserverless',
        'GetAccessPolicy',
        'get_access_policy',
        {'type': 'data', 'name': POLICY_NAME},
        'ResourceNotFoundException',
        True,
        id='serverless_not_found',
    ),
    pytest.param(
        'opensearchserverless',
        'CreateIndex',
        'create_index',
        {'id': COLLECTION_ID, 'indexName': MEMORY_INDEX, 'indexSchema': SCHEMA},
        'ConflictException',
        True,
        id='serverless_conflict',
    ),
    pytest.param(
        'opensearch',
        'CreateIndex',
        'create_index',
        {'DomainName': DOMAIN, 'IndexName': MEMORY_INDEX, 'IndexSchema': SCHEMA},
        'ResourceAlreadyExistsException',
        True,
        id='domain_already_exists',
    ),
    pytest.param(
        'sts',
        'GetCallerIdentity',
        'get_caller_identity',
        {},
        'ExpiredToken',
        False,
        id='sts_expired_token',
    ),
]


@pytest.mark.parametrize('service, operation, method, arguments, code, modeled', ERRORS)
def test_error_reply_raises_as_botocore_does(
    boto3_stub,
    credentials,
    http_sends,
    golden,
    normalize,
    request,
    service,
    operation,
    method,
    arguments,
    code,
    modeled,
):
    boto3_stub.error(service, operation, code)
    stubbed = client(service)

    with pytest.raises(ClientError) as raised:
        getattr(stubbed, method)(**arguments)

    error = raised.value
    # A modeled code raises the client's own exception class, as from the real service.
    if modeled:
        assert isinstance(error, getattr(stubbed.exceptions, code))
    assert type(error).__name__ == (code if modeled else 'ClientError')
    assert error.response['Error']['Code'] == code
    assert len(boto3_stub.calls) == 1
    assert http_sends == []
    golden(
        f'{AREA}/error_{request.node.callspec.id}',
        normalize(
            {
                'call': [service, operation, arguments],
                'exception': type(error).__name__,
                'message': str(error),
                'response': error.response,
                'calls': boto3_stub.calls,
            }
        ),
    )


def test_error_status_defaults_to_the_modeled_status(boto3_stub, credentials):
    boto3_stub.error('opensearch', 'CreateIndex', 'DependencyFailureException')
    boto3_stub.error('opensearchserverless', 'CreateIndex', 'ConflictException', status=409)

    with pytest.raises(ClientError) as domain:
        client('opensearch').create_index(DomainName=DOMAIN, IndexName='a', IndexSchema={})
    with pytest.raises(ClientError) as serverless:
        client('opensearchserverless').create_index(id=COLLECTION_ID, indexName='a')

    assert domain.value.response['ResponseMetadata']['HTTPStatusCode'] == 424
    assert serverless.value.response['ResponseMetadata']['HTTPStatusCode'] == 409


def test_newest_reply_wins(boto3_stub, credentials):
    boto3_stub.reply('opensearch', 'CreateIndex', {'Status': 'CREATED'})
    boto3_stub.error('opensearch', 'CreateIndex', 'ResourceAlreadyExistsException')
    domain = client('opensearch')

    with pytest.raises(domain.exceptions.ResourceAlreadyExistsException):
        domain.create_index(DomainName=DOMAIN, IndexName=MEMORY_INDEX, IndexSchema={})


def test_call_with_no_reply_fails_clearly(harness, boto3_stub, credentials, http_sends):
    boto3_stub.reply('opensearchserverless', 'BatchGetCollection', {'collectionDetails': []})

    with pytest.raises(
        harness.UnmatchedBoto3CallError,
        match='no canned reply for opensearchserverless CreateIndex',
    ):
        client('opensearchserverless').create_index(id=COLLECTION_ID, indexName=MEMORY_INDEX)

    record = {
        'service': 'opensearchserverless',
        'operation': 'CreateIndex',
        'params': {'id': COLLECTION_ID, 'indexName': MEMORY_INDEX},
        'host': f'aoss.{REGION}.amazonaws.com',
        'auth': {'kind': 'sigv4', 'region': REGION, 'service': 'aoss'},
    }
    # Logged as any call, never answered, and the test would fail at teardown.
    assert boto3_stub.calls == boto3_stub.unmatched == [record]
    assert http_sends == []
    with pytest.raises(pytest.fail.Exception, match='boto3 stub received calls with no canned'):
        boto3_stub.verify()
    boto3_stub.unmatched.clear()


def test_botocore_still_validates_parameters(boto3_stub, credentials, http_sends):
    boto3_stub.reply('opensearch', 'CreateIndex', {'Status': 'CREATED'})
    boto3_stub.reply('opensearchserverless', 'CreateAccessPolicy', {})

    # The camelCase names the memory index auto-create passes to the `opensearch` API.
    with pytest.raises(ParamValidationError, match='Unknown parameter in input: "domainName"'):
        client('opensearch').create_index(
            domainName=DOMAIN, indexName=MEMORY_INDEX, indexSchema=json.dumps(SCHEMA)
        )
    with pytest.raises(ParamValidationError, match='valid min length: 3'):
        client('opensearchserverless').create_access_policy(type='data', name='x', policy='[]')

    # Rejected before signing and sending: the stub never saw them.
    assert boto3_stub.calls == []
    assert http_sends == []


def test_botocore_still_signs(boto3_stub, http_sends, monkeypatch):
    boto3_stub.reply('sts', 'GetCallerIdentity', SUPPORTED[0][4])

    # No credentials: botocore cannot sign, so the call fails before it is sent.
    with pytest.raises(NoCredentialsError):
        client('sts').get_caller_identity()
    assert boto3_stub.calls == []

    for name, value in boto3_stub.env.items():
        monkeypatch.setenv(name, value)
    client('sts').get_caller_identity()
    assert [(c['host'], c['auth']) for c in boto3_stub.calls] == [
        (f'sts.{REGION}.amazonaws.com', {'kind': 'sigv4', 'region': REGION, 'service': 'sts'})
    ]
    assert http_sends == []


def test_every_botocore_client_is_stubbed(boto3_stub, credentials, http_sends):
    # boto3's default session, a boto3 session and a bare botocore session alike.
    boto3_stub.reply('sts', 'GetCallerIdentity', SUPPORTED[0][4])
    clients = [
        boto3.client('sts', region_name=REGION),
        boto3.Session(region_name=REGION).client('sts'),
        botocore.session.get_session().create_client('sts', region_name=REGION),
    ]

    for sts in clients:
        assert sts.get_caller_identity()['Arn'] == SESSION_ARN
    assert len(boto3_stub.calls) == len(clients)
    assert http_sends == []


def test_reply_must_name_an_operation_and_its_output(boto3_stub):
    with pytest.raises(ValueError, match='no botocore operation aoss CreateIndex'):
        boto3_stub.reply('aoss', 'CreateIndex')
    with pytest.raises(ValueError, match='no botocore operation sts GetCallerIdentityy'):
        boto3_stub.error('sts', 'GetCallerIdentityy', 'ExpiredToken')
    with pytest.raises(ValueError, match=r"\['arn'\] are not members of its output"):
        boto3_stub.reply('sts', 'GetCallerIdentity', {'arn': SESSION_ARN})
    with pytest.raises(TypeError, match='give the data as JSON data'):
        boto3_stub.reply('opensearch', 'CreateIndex', {'Status': object()})


def test_replies_cross_the_launch_spec(harness, boto3_stub):
    for service, operation, _, _, data in SUPPORTED:
        boto3_stub.reply(service, operation, data)
    boto3_stub.error('opensearchserverless', 'GetAccessPolicy', 'ResourceNotFoundException')

    spec = json.loads(json.dumps(boto3_stub.spec()))

    assert harness.Boto3Stub.from_spec(spec)._replies == boto3_stub._replies


def test_without_the_fixture_nothing_is_stubbed(harness):
    real = botocore.session.Session.create_client
    with pytest.MonkeyPatch.context() as monkeypatch:
        harness.Boto3Stub().install(monkeypatch.setattr)
        assert botocore.session.Session.create_client is not real

    assert botocore.session.Session.create_client is real


# --- through the launched server -------------------------------------------------

NO_AUTH = {'OPENSEARCH_NO_AUTH': 'true'}
SERVERLESS_ENV = {
    **NO_AUTH,
    'AWS_OPENSEARCH_SERVERLESS': 'true',
    'AWS_OPENSEARCH_COLLECTION_ID': COLLECTION_ID,
    'AWS_REGION': REGION,
}
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


def first_save_routes(fake_opensearch) -> None:
    """The memory index is missing (so the save creates it), then the document is indexed."""
    fake_opensearch.reply('HEAD', f'/{MEMORY_INDEX}', None, status=404)
    fake_opensearch.reply('POST', f'/{MEMORY_INDEX}/_doc', INDEXED, status=201)


async def test_launched_server_is_answered_by_the_stub(
    memory_enabled_session, boto3_stub, fake_opensearch, no_network, golden, normalize
):
    first_save_routes(fake_opensearch)
    boto3_stub.reply('sts', 'GetCallerIdentity', SUPPORTED[0][4])
    boto3_stub.reply('opensearchserverless', 'BatchGetCollection', SUPPORTED[1][4])
    boto3_stub.error('opensearchserverless', 'GetAccessPolicy', 'ResourceNotFoundException')
    boto3_stub.reply('opensearchserverless', 'CreateAccessPolicy', SUPPORTED[3][4])
    boto3_stub.reply('opensearchserverless', 'CreateIndex')
    async with memory_enabled_session(env={**SERVERLESS_ENV, **boto3_stub.env}) as session:
        result = await session.call_tool('SaveMemoryTool', SAVE_ARGS)
        logged = session.boto3_calls()

    # The launched server's calls, as its launcher logged them, joined the stub's on exit.
    assert logged == boto3_stub.calls
    assert [c['operation'] for c in boto3_stub.calls] == [
        'GetCallerIdentity',
        'BatchGetCollection',
        'GetAccessPolicy',
        'CreateAccessPolicy',
        'CreateIndex',
    ]
    assert boto3_stub.unmatched == []
    assert no_network == []
    golden(
        f'{AREA}/launched_serverless_first_save',
        normalize(
            {
                'call': {'name': 'SaveMemoryTool', 'arguments': SAVE_ARGS},
                'result': result,
                'boto3_calls': boto3_stub.calls,
                'requests': fake_opensearch.requests,
            }
        ),
    )


async def test_launched_server_call_with_no_reply_fails_clearly(
    memory_enabled_session, boto3_stub, fake_opensearch, no_network, golden, normalize
):
    fake_opensearch.reply('HEAD', f'/{MEMORY_INDEX}', None, status=404)
    boto3_stub.reply('sts', 'GetCallerIdentity', SUPPORTED[0][4])
    async with memory_enabled_session(env={**SERVERLESS_ENV, **boto3_stub.env}) as session:
        result = await session.call_tool('SaveMemoryTool', SAVE_ARGS)

    assert result['isError'] is True
    assert 'no canned reply for opensearchserverless BatchGetCollection' in str(result)
    assert [c['operation'] for c in boto3_stub.unmatched] == ['BatchGetCollection']
    assert no_network == []
    golden(
        f'{AREA}/launched_no_reply',
        normalize(
            {
                'result': result,
                'boto3_calls': boto3_stub.calls,
                'unmatched': boto3_stub.unmatched,
            }
        ),
    )
    # Shown above; without this the test fails at teardown, as an unexpected call would.
    boto3_stub.unmatched.clear()


async def test_launched_server_without_the_fixture_is_not_stubbed(harness, stdio_launcher):
    async with stdio_launcher(env=NO_AUTH) as server:
        await server.initialize()
        with open(server.log_path.with_name(harness.LAUNCH_SPEC_FILE), encoding='utf-8') as f:
            spec = json.load(f)

    assert 'boto3_stub' not in spec
    assert server.boto3_calls() == []

import pytest
import tempfile
import os
import yaml
import platform
from unittest.mock import patch, MagicMock
from mcp_server_opensearch.clusters_information import (
    ClusterInfo,
    add_cluster,
    get_cluster,
    load_clusters_from_yaml,
    check_cluster_connection,
    cluster_registry,
)


class TestClusterInfo:
    def test_cluster_info_creation_with_required_fields(self):
        """Test creating ClusterInfo with only required fields."""
        cluster = ClusterInfo(opensearch_url='https://localhost:9200')
        assert cluster.opensearch_url == 'https://localhost:9200'
        assert cluster.iam_arn is None
        assert cluster.aws_region is None
        assert cluster.opensearch_username is None
        assert cluster.opensearch_password is None
        assert cluster.profile is None

    def test_cluster_info_creation_with_all_fields(self):
        """Test creating ClusterInfo with all fields."""
        cluster = ClusterInfo(
            opensearch_url='https://localhost:9200',
            iam_arn='arn:aws:iam::123456789012:role/OpenSearchRole',
            aws_region='us-west-2',
            opensearch_username='admin',
            opensearch_password='password123',
            profile='default',
        )
        assert cluster.opensearch_url == 'https://localhost:9200'
        assert cluster.iam_arn == 'arn:aws:iam::123456789012:role/OpenSearchRole'
        assert cluster.aws_region == 'us-west-2'
        assert cluster.opensearch_username == 'admin'
        assert cluster.opensearch_password == 'password123'
        assert cluster.profile == 'default'

    def test_cluster_info_validation(self):
        """Test that ClusterInfo validates required fields."""
        with pytest.raises(ValueError):
            ClusterInfo()  # Missing required opensearch_url


class TestClusterRegistry:
    def setup_method(self):
        """Clear the cluster registry before each test."""
        cluster_registry.clear()

    def test_add_cluster(self):
        """Test adding a cluster to the registry."""
        cluster = ClusterInfo(opensearch_url='https://localhost:9200')
        add_cluster('test-cluster', cluster)

        assert 'test-cluster' in cluster_registry
        assert cluster_registry['test-cluster'] == cluster

    def test_add_cluster_overwrites_existing(self):
        """Test that adding a cluster with existing name overwrites it."""
        cluster1 = ClusterInfo(opensearch_url='https://localhost:9200')
        cluster2 = ClusterInfo(opensearch_url='https://localhost:9201')

        add_cluster('test-cluster', cluster1)
        add_cluster('test-cluster', cluster2)

        assert cluster_registry['test-cluster'] == cluster2
        assert len(cluster_registry) == 1

    def test_get_cluster_existing(self):
        """Test getting an existing cluster."""
        cluster = ClusterInfo(opensearch_url='https://localhost:9200')
        add_cluster('test-cluster', cluster)

        retrieved = get_cluster('test-cluster')
        assert retrieved == cluster

    def test_get_cluster_nonexistent(self):
        """Test getting a non-existent cluster returns None."""
        retrieved = get_cluster('nonexistent-cluster')
        assert retrieved is None


class TestLoadClustersFromYaml:
    def setup_method(self):
        """Clear the cluster registry before each test."""
        cluster_registry.clear()

    def test_load_clusters_from_yaml_empty_file(self):
        """Test loading from empty YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write('clusters: {}')
            f.flush()

            load_clusters_from_yaml(f.name)

        os.unlink(f.name)
        assert len(cluster_registry) == 0

    def test_load_clusters_from_yaml_valid_clusters(self):
        """Test loading valid clusters from YAML."""
        yaml_content = """
clusters:
  cluster1:
    opensearch_url: "https://localhost:9200"
    opensearch_username: "admin"
    opensearch_password: "password"
  cluster2:
    opensearch_url: "https://localhost:9201"
    iam_arn: "arn:aws:iam::123456789012:role/OpenSearchRole"
    aws_region: "us-west-2"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            with patch(
                'mcp_server_opensearch.clusters_information.check_cluster_connection'
            ) as mock_check:
                mock_check.return_value = (True, '')
                load_clusters_from_yaml(f.name)

        os.unlink(f.name)

        assert len(cluster_registry) == 2
        assert 'cluster1' in cluster_registry
        assert 'cluster2' in cluster_registry

        cluster1 = cluster_registry['cluster1']
        assert cluster1.opensearch_url == 'https://localhost:9200'
        assert cluster1.opensearch_username == 'admin'
        assert cluster1.opensearch_password == 'password'

        cluster2 = cluster_registry['cluster2']
        assert cluster2.opensearch_url == 'https://localhost:9201'
        assert cluster2.iam_arn == 'arn:aws:iam::123456789012:role/OpenSearchRole'
        assert cluster2.aws_region == 'us-west-2'

    def test_load_clusters_from_yaml_missing_opensearch_url(self):
        """Test loading cluster without required opensearch_url."""
        yaml_content = """
clusters:
  invalid_cluster:
    opensearch_username: "admin"
    opensearch_password: "password"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            with patch(
                'mcp_server_opensearch.clusters_information.check_cluster_connection'
            ) as mock_check:
                mock_check.return_value = (True, '')
                load_clusters_from_yaml(f.name)

        os.unlink(f.name)

        # Cluster should not be added due to missing opensearch_url
        assert len(cluster_registry) == 0

    def test_load_clusters_from_yaml_connection_failure(self):
        """Test loading cluster that fails connection check."""
        yaml_content = """
clusters:
  unreachable_cluster:
    opensearch_url: "https://unreachable:9200"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            with patch(
                'mcp_server_opensearch.clusters_information.check_cluster_connection'
            ) as mock_check:
                mock_check.return_value = (False, 'Connection timeout')
                load_clusters_from_yaml(f.name)

        os.unlink(f.name)

        # Cluster should not be added due to connection failure
        assert len(cluster_registry) == 0

    def test_load_clusters_from_yaml_file_not_found(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_clusters_from_yaml('/nonexistent/file.yml')

    def test_load_clusters_from_yaml_invalid_yaml(self):
        """Test loading from invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write('invalid: yaml: content: [')
            f.flush()

            with pytest.raises(yaml.YAMLError):
                load_clusters_from_yaml(f.name)

        os.unlink(f.name)

    def test_load_clusters_from_yaml_permission_error(self):
        """Test loading from file with permission error."""
        # Skip this test on Windows as os.chmod with 0o000 doesn't work reliably
        if platform.system() == 'Windows':
            pytest.skip('Permission test not reliable on Windows')

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write('clusters: {}')
            f.flush()
            os.chmod(f.name, 0o000)  # Remove all permissions

            with pytest.raises(PermissionError):
                load_clusters_from_yaml(f.name)

        os.chmod(f.name, 0o644)  # Restore permissions
        os.unlink(f.name)

    def test_load_clusters_from_yaml_empty_path(self):
        """Test loading from empty path."""
        load_clusters_from_yaml('')
        assert len(cluster_registry) == 0

    def test_load_clusters_from_yaml_none_path(self):
        """Test loading from None path."""
        load_clusters_from_yaml(None)
        assert len(cluster_registry) == 0


class TestCheckClusterConnection:
    def test_check_cluster_connection_success(self):
        """Test successful cluster connection."""
        cluster = ClusterInfo(opensearch_url='https://localhost:9200')

        with patch('opensearch.client.initialize_client_with_cluster') as mock_init:
            mock_client = MagicMock()
            mock_client.info.return_value = {'version': {'number': '2.0.0'}}
            mock_init.return_value = mock_client

            success, error = check_cluster_connection(cluster)

            assert success is True
            assert error == ''
            mock_client.info.assert_called_once()

    def test_check_cluster_connection_failure(self):
        """Test failed cluster connection."""
        cluster = ClusterInfo(opensearch_url='https://unreachable:9200')

        with patch('opensearch.client.initialize_client_with_cluster') as mock_init:
            mock_init.side_effect = Exception('Connection timeout')

            success, error = check_cluster_connection(cluster)

            assert success is False
            assert 'Connection timeout' in error

    def test_check_cluster_connection_client_info_failure(self):
        """Test cluster connection where client.info() fails."""
        cluster = ClusterInfo(opensearch_url='https://localhost:9200')

        with patch('opensearch.client.initialize_client_with_cluster') as mock_init:
            mock_client = MagicMock()
            mock_client.info.side_effect = Exception('Authentication failed')
            mock_init.return_value = mock_client

            success, error = check_cluster_connection(cluster)

            assert success is False
            assert 'Authentication failed' in error

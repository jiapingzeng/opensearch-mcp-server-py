import pytest
from semver import Version
from unittest.mock import patch, MagicMock
from tools.utils import is_tool_compatible
from tools.tool_filter import get_tools
from tools.tool_params import baseToolArgs


class TestIsToolCompatible:
    def test_version_within_range(self):
        tool_info = {'min_version': '1.0.0', 'max_version': '3.0.0'}
        assert is_tool_compatible(Version.parse('2.0.0'), tool_info) is True

    def test_version_below_min(self):
        tool_info = {'min_version': '2.0.0', 'max_version': '3.0.0'}
        assert is_tool_compatible(Version.parse('1.5.0'), tool_info) is False

    def test_version_above_max(self):
        tool_info = {'min_version': '1.0.0', 'max_version': '2.0.0'}
        assert is_tool_compatible(Version.parse('2.1.0'), tool_info) is False

    def test_version_equal_to_min(self):
        tool_info = {'min_version': '1.0.0', 'max_version': '3.0.0'}
        assert is_tool_compatible(Version.parse('1.0.0'), tool_info) is True

    def test_version_equal_to_max(self):
        tool_info = {'min_version': '1.0.0', 'max_version': '3.0.0'}
        assert is_tool_compatible(Version.parse('3.0.0'), tool_info) is True

    def test_version_only_patch_not_provided(self):
        tool_info = {'min_version': '2.5', 'max_version': '3'}
        assert is_tool_compatible(Version.parse('2.5.1'), tool_info) is True
        assert is_tool_compatible(Version.parse('2.15.0'), tool_info) is True
        assert is_tool_compatible(Version.parse('3.0.0'), tool_info) is True

    def test_default_tool_info(self):
        # Should be True for almost any reasonable version
        assert is_tool_compatible(Version.parse('1.2.3')) is True
        assert is_tool_compatible(Version.parse('99.0.0')) is True
        assert is_tool_compatible(Version.parse('0.0.1')) is True

    def test_invalid_version_strings(self):
        # If min_version or max_version is not a valid semver, should raise ValueError
        with pytest.raises(ValueError):
            is_tool_compatible(Version.parse('1.0.0'), {'min_version': 'not_a_version'})
        with pytest.raises(ValueError):
            is_tool_compatible(Version.parse('1.0.0'), {'max_version': 'not_a_version'})


class TestGetTools:
    @pytest.fixture
    def mock_tool_registry(self):
        """Shared fixture for mock tool registry data."""
        return {
            'ListIndexTool': {
                'description': 'List indices',
                'input_schema': {
                    'type': 'object',
                    'title': 'ListIndexArgs',
                    'properties': {
                        'opensearch_cluster_name': {'type': 'string'},
                        'index': {'type': 'string'},
                        'custom_field': {'type': 'string'},
                    },
                    'required': ['index'],
                },
                'function': MagicMock(),
                'args_model': MagicMock(),
                'min_version': '1.0.0',
                'max_version': '3.0.0',
            },
            'SearchIndexTool': {
                'description': 'Search index',
                'input_schema': {
                    'type': 'object',
                    'properties': {
                        'opensearch_cluster_name': {'type': 'string'},
                        'query': {'type': 'object'},
                    },
                },
                'function': MagicMock(),
                'args_model': MagicMock(),
                'min_version': '2.0.0',
                'max_version': '4.0.0',
            },
        }

    @pytest.fixture
    def mock_patches(self):
        """Shared fixture for common mock patches."""
        return [
            patch('tools.tool_filter.get_opensearch_version'),
            patch('tools.tool_filter.TOOL_REGISTRY'),
            patch('tools.tool_filter.is_tool_compatible'),
        ]

    def test_get_tools_multi_mode_returns_all_tools(self, mock_tool_registry):
        """Test that multi mode returns all tools with base fields intact."""
        with patch('tools.tool_filter.TOOL_REGISTRY', new=mock_tool_registry):
            result = get_tools(mode='multi')

            assert len(result) == 2
            assert 'ListIndexTool' in result
            assert 'SearchIndexTool' in result

            # Base fields should be present in multi mode
            assert (
                'opensearch_cluster_name' in result['ListIndexTool']['input_schema']['properties']
            )
            assert (
                'opensearch_cluster_name'
                in result['SearchIndexTool']['input_schema']['properties']
            )

    def test_get_tools_single_mode_filters_and_removes_base_fields(
        self, mock_tool_registry, mock_patches
    ):
        """Test that single mode filters by version AND removes base fields."""
        mock_get_version, mock_tool_reg, mock_is_compatible = mock_patches

        with (
            mock_get_version as mock_version,
            mock_tool_reg as mock_reg,
            mock_is_compatible as mock_compat,
        ):
            # Setup mocks
            mock_version.return_value = Version.parse('2.5.0')
            mock_reg.items.return_value = mock_tool_registry.items()

            # Mock compatibility: only ListIndexTool should be compatible
            mock_compat.side_effect = (
                lambda version, tool_info: tool_info['min_version'] == '1.0.0'
            )

            # Call get_tools in single mode
            result = get_tools(mode='single')

            # Should only return compatible tools
            assert len(result) == 1
            assert 'ListIndexTool' in result
            assert 'SearchIndexTool' not in result

            # Should remove base fields from schema
            schema = result['ListIndexTool']['input_schema']
            assert 'opensearch_cluster_name' not in schema['properties']
            assert 'index' in schema['properties']
            assert 'custom_field' in schema['properties']

            # Should preserve schema structure
            assert schema['type'] == 'object'
            assert schema['title'] == 'ListIndexArgs'
            assert 'required' in schema
            assert 'index' in schema['required']

    @patch.dict('os.environ', {'AWS_OPENSEARCH_SERVERLESS': 'true'})
    def test_get_tools_single_mode_serverless_skips_version_check(
        self, mock_tool_registry, mock_patches
    ):
        """Test that serverless mode skips version compatibility checks."""
        mock_get_version, mock_tool_reg, mock_is_compatible = mock_patches

        with (
            mock_get_version as mock_version,
            mock_tool_reg as mock_reg,
            mock_is_compatible as mock_compat,
        ):
            # Setup mocks
            mock_version.return_value = Version.parse('2.5.0')
            mock_reg.items.return_value = mock_tool_registry.items()

            # Call get_tools in single mode with serverless environment
            result = get_tools(mode='single')

            # Should return all tools despite version incompatibility
            assert len(result) == 2
            assert 'ListIndexTool' in result
            assert 'SearchIndexTool' in result

            # Version compatibility check should not be called
            mock_compat.assert_not_called()

    def test_get_tools_single_mode_handles_missing_properties(self, mock_patches):
        """Test that single mode handles schemas without properties field."""
        mock_get_version, mock_tool_reg, mock_is_compatible = mock_patches

        # Create tool with missing properties
        tool_without_properties = {
            'ListIndexTool': {
                'description': 'List indices',
                'input_schema': {
                    'type': 'object',
                    'title': 'ListIndexArgs',
                    # No properties field
                },
                'function': MagicMock(),
                'args_model': MagicMock(),
                'min_version': '1.0.0',
                'max_version': '3.0.0',
            }
        }

        with (
            mock_get_version as mock_version,
            mock_tool_reg as mock_reg,
            mock_is_compatible as mock_compat,
        ):
            mock_version.return_value = Version.parse('2.5.0')
            mock_compat.return_value = True
            mock_reg.items.return_value = tool_without_properties.items()

            # Call get_tools in single mode - should not raise error
            result = get_tools(mode='single')

            assert len(result) == 1
            assert 'ListIndexTool' in result

    def test_get_tools_default_mode_is_single(self, mock_patches):
        """Test that get_tools defaults to single mode."""
        mock_get_version, mock_tool_reg, mock_is_compatible = mock_patches

        with (
            mock_get_version as mock_version,
            mock_tool_reg as mock_reg,
            mock_is_compatible as mock_compat,
        ):
            mock_version.return_value = Version.parse('2.5.0')
            mock_compat.return_value = True
            mock_reg.items.return_value = []

            # Call get_tools without specifying mode
            get_tools()

            # Should call get_opensearch_version (single mode behavior)
            mock_version.assert_called_once()

    def test_get_tools_logs_version_info(self, mock_patches, caplog):
        """Test that get_tools logs version information in single mode."""
        mock_get_version, mock_tool_reg, mock_is_compatible = mock_patches

        with (
            mock_get_version as mock_version,
            mock_tool_reg as mock_reg,
            mock_is_compatible as mock_compat,
        ):
            mock_version.return_value = Version.parse('2.5.0')
            mock_compat.return_value = True
            mock_reg.items.return_value = []

            # Call get_tools in single mode with logging capture
            with caplog.at_level('INFO'):
                get_tools(mode='single')

            # Should log version information
            assert 'Connected OpenSearch version: 2.5.0' in caplog.text


class TestProcessToolFilter:
    def setup_method(self):
        """Setup that runs before each test method."""
        # Tool registry with http methods
        tool_registry = {
            'ListIndexTool': {'http_methods': 'GET'},
            'ClusterHealthTool': {'http_methods': 'GET'},
            'SearchIndexTool': {'http_methods': 'GET, POST'},
            'IndicesCreateTool': {'http_methods': 'PUT'},
            'ExplainTool': {'http_methods': 'GET, POST'},
            'MsearchTool': {'http_methods': 'GET, POST'},
        }

        # Tool registry (case insensitive)
        tool_registry_lower = {
            'listindextool': 'ListIndexTool',
            'searchindextool': 'SearchIndexTool',
            'getmappingtool': 'GetMappingTool',
        }

        from tools.tool_filter import (
            parse_comma_separated,
            load_yaml_config,
            process_categories,
            process_regex_patterns,
            validate_tools,
            apply_write_filter,
            process_tool_filter,
        )

        self.tool_registry = tool_registry
        self.tool_registry_lower = tool_registry_lower
        self.parse_comma_separated = parse_comma_separated
        self.load_yaml_config = load_yaml_config
        self.process_categories = process_categories
        self.process_regex_patterns = process_regex_patterns
        self.validate_tools = validate_tools
        self.apply_write_filter = apply_write_filter
        self.process_tool_filter = process_tool_filter

    def test_parse_comma_separated(self):
        """Test that comma separated values are parsed correctly."""
        assert self.parse_comma_separated('a,b,c') == ['a', 'b', 'c']
        assert self.parse_comma_separated('a') == ['a']
        assert self.parse_comma_separated('') == []
        assert self.parse_comma_separated(None) == []

    def test_load_valid_yaml_config(self):
        """Test loading a valid yaml config."""
        config = self.load_yaml_config('tests/tools/test_config.yml')
        assert config == {
            'tool_category': {'critical': ['SearchIndexTool', 'ExplainTool']},
            'tool_filters': {
                'disabled_categories': ['critical'],
                'disabled_tools': ['MsearchTool'],
            },
        }

    def test_load_invalid_yaml_config(self, caplog):
        """Test loading an invalid yaml config."""
        config = self.load_yaml_config('tests/tools/test_invalid_config.yml')
        assert config is None
        assert 'Error loading filter config' in caplog.text

    def test_process_valid_categories(self):
        """Test that valid categories are processed correctly."""
        config = self.load_yaml_config('tests/tools/test_config.yml')
        tool_category = config.get('tool_category', {})
        categories = ['critical']
        process_categories = self.process_categories(categories, tool_category)
        assert process_categories == ['SearchIndexTool', 'ExplainTool']

    def test_process_invalid_categories(self, caplog):
        """Test processing invalid category."""
        config = self.load_yaml_config('tests/tools/test_config.yml')
        tool_category = config.get('tool_category', {})
        categories = ['invalid']
        process_categories = self.process_categories(categories, tool_category)
        assert process_categories == []
        assert "Category 'invalid' not found in tool categories" in caplog.text

    def test_process_regex_patterns(self):
        """Test processing regex patterns."""
        regex_patterns = ['search.*', 'Get.*']
        tool_names = ['ListIndexTool', 'GetMappingTool', 'SearchTool', 'DeleteTool']
        matching_tools = self.process_regex_patterns(regex_patterns, tool_names)

        assert 'ListIndexTool' not in matching_tools  # Doesn't match either pattern
        assert 'GetMappingTool' in matching_tools  # Matches Get.*
        assert 'SearchTool' in matching_tools  # Matches search.*
        assert 'DeleteTool' not in matching_tools  # Doesn't match either pattern
        assert len(matching_tools) == 2

    def test_validate_valid_tools(self):
        """Test validating valid tools."""
        tool_list = ['ListIndexTool', 'SearchIndexTool']
        source_name = 'disabled_tools'
        valid_tools = self.validate_tools(tool_list, self.tool_registry_lower, source_name)
        assert 'listindextool' in valid_tools
        assert 'searchindextool' in valid_tools

        # Test case insensitivity
        tool_list = ['GetMappingTool', 'SEARCHINDEXTOOL']
        valid_tools = self.validate_tools(tool_list, self.tool_registry_lower, source_name)
        assert 'getmappingtool' in valid_tools
        assert 'searchindextool' in valid_tools

    def test_validate_invalid_tools(self, caplog):
        """Test validating invalid tools."""
        tool_list = ['InvalidTool', 'exampletool']
        source_name = 'disabled_tools'
        valid_tools = self.validate_tools(tool_list, self.tool_registry_lower, source_name)
        assert valid_tools == set()
        assert "Ignoring invalid tool from 'disabled_tools': 'InvalidTool'" in caplog.text
        assert "Ignoring invalid tool from 'disabled_tools': 'exampletool'" in caplog.text

    def test_apply_write_filter(self, monkeypatch):
        """Test that apply write filter are applied correctly."""
        tool_registry_copy = self.tool_registry.copy()

        monkeypatch.setenv('OPENSEARCH_SETTINGS_ALLOW_WRITE', False)
        self.apply_write_filter(tool_registry_copy)
        assert 'ListIndexTool' in tool_registry_copy
        assert 'SearchIndexTool' in tool_registry_copy
        assert 'IndicesCreateTool' not in tool_registry_copy

    def test_process_tool_filter_config(self, monkeypatch, caplog):
        """Test processing tool filter from a YAML config file."""
        # Set the log level to capture all messages
        import logging

        caplog.set_level(logging.INFO)

        # Patch the TOOL_REGISTRY in the tool_filter module
        tool_registry_copy = self.tool_registry.copy()
        monkeypatch.setattr('tools.tool_filter.TOOL_REGISTRY', tool_registry_copy)

        # Call the function with the config file
        self.process_tool_filter(filter_path='tests/tools/test_config.yml')

        # Check the results
        assert 'ClusterHealthTool' in tool_registry_copy
        assert 'ListIndexTool' in tool_registry_copy
        assert 'MsearchTool' not in tool_registry_copy  # In disabled_tools
        assert 'SearchIndexTool' not in tool_registry_copy  # In critical category
        assert 'ExplainTool' not in tool_registry_copy  # In critical category

        # Check that the right log messages were produced
        assert 'Applied tool filter from tests/tools/test_config.yml' in caplog.text

    def test_process_tool_filter_env(self, monkeypatch, caplog):
        """Test processing tool filter from environment variables."""
        # Set the log level to capture all messages
        import logging

        caplog.set_level(logging.INFO)

        # Patch the TOOL_REGISTRY in the tool_filter module
        tool_registry_copy = self.tool_registry.copy()
        monkeypatch.setattr('tools.tool_filter.TOOL_REGISTRY', tool_registry_copy)

        # Set environment variables via monkeypatch
        monkeypatch.setenv('OPENSEARCH_DISABLED_TOOLS', 'ExplainTool')
        monkeypatch.setenv('OPENSEARCH_DISABLED_TOOLS_REGEX', 'search.*')
        monkeypatch.setenv('OPENSEARCH_SETTINGS_ALLOW_WRITE', True)

        # Call the function with environment variables
        self.process_tool_filter(
            disabled_tools='ExplainTool', disabled_tools_regex='search.*', allow_write=True
        )

        # Check the results
        assert 'ListIndexTool' in tool_registry_copy
        assert 'ClusterHealthTool' in tool_registry_copy
        assert 'IndicesCreateTool' in tool_registry_copy
        assert 'MsearchTool' in tool_registry_copy
        assert 'SearchIndexTool' not in tool_registry_copy  # In disabled_tools_regex
        assert 'ExplainTool' not in tool_registry_copy  # In disabled_tools

        # Check that the right log messages were produced
        assert 'Applied tool filter from environment variables' in caplog.text

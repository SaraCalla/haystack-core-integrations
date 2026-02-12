# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


import pytest
from haystack.errors import FilterError

from haystack_integrations.document_stores.vespa.filters import normalize_filters

filters_data = [
    # Complex AND with nested OR, date ranges, and numeric range
    (
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.type", "operator": "==", "value": "article"},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.genre", "operator": "in", "value": ["economy", "politics"]},
                        {"field": "meta.publisher", "operator": "==", "value": "nytimes"},
                    ],
                },
                {"field": "meta.date", "operator": ">=", "value": "2015-01-01"},
                {"field": "meta.date", "operator": "<", "value": "2021-01-01"},
                {"field": "meta.rating", "operator": ">=", "value": 3},
            ],
        },
        '(type contains "article"'
        " and (genre in ('economy', 'politics')"
        ' or publisher contains "nytimes")'
        ' and date >= "2015-01-01"'
        ' and date < "2021-01-01"'
        " and rating >= 3)",
    ),
    # OR of two AND groups
    (
        {
            "operator": "OR",
            "conditions": [
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.Type", "operator": "==", "value": "News Paper"},
                        {"field": "meta.Date", "operator": "<", "value": "2020-01-01"},
                    ],
                },
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.Type", "operator": "==", "value": "Blog Post"},
                        {"field": "meta.Date", "operator": ">=", "value": "2019-01-01"},
                    ],
                },
            ],
        },
        '((Type contains "News Paper" and Date < "2020-01-01")'
        ' or (Type contains "Blog Post" and Date >= "2019-01-01"))',
    ),
    # AND with reordered conditions (OR at the end)
    (
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.type", "operator": "==", "value": "article"},
                {"field": "meta.date", "operator": ">=", "value": "2015-01-01"},
                {"field": "meta.date", "operator": "<", "value": "2021-01-01"},
                {"field": "meta.rating", "operator": ">=", "value": 3},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.genre", "operator": "in", "value": ["economy", "politics"]},
                        {"field": "meta.publisher", "operator": "==", "value": "nytimes"},
                    ],
                },
            ],
        },
        '(type contains "article"'
        ' and date >= "2015-01-01"'
        ' and date < "2021-01-01"'
        " and rating >= 3"
        " and (genre in ('economy', 'politics')"
        ' or publisher contains "nytimes"))',
    ),
    # Text field equality (no special handling in Vespa, uses contains like any string)
    (
        {"operator": "AND", "conditions": [{"field": "text", "operator": "==", "value": "A Foo Document 1"}]},
        '(text contains "A Foo Document 1")',
    ),
    # Nested OR with range
    (
        {
            "operator": "OR",
            "conditions": [
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.name", "operator": "==", "value": "name_0"},
                        {"field": "meta.name", "operator": "==", "value": "name_1"},
                    ],
                },
                {"field": "meta.number", "operator": "<", "value": 1.0},
            ],
        },
        '((name contains "name_0" or name contains "name_1") or number < 1.0)',
    ),
    # Range operators with in
    (
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.number", "operator": "<=", "value": 2},
                {"field": "meta.number", "operator": ">=", "value": 0},
                {"field": "meta.name", "operator": "in", "value": ["name_0", "name_1"]},
            ],
        },
        "(number <= 2 and number >= 0 and name in ('name_0', 'name_1'))",
    ),
    # Two ranges on same field (no merging needed in Vespa unlike OpenSearch)
    (
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.number", "operator": "<=", "value": 2},
                {"field": "meta.number", "operator": ">=", "value": 0},
            ],
        },
        "(number <= 2 and number >= 0)",
    ),
    # Simple OR
    (
        {
            "operator": "OR",
            "conditions": [
                {"field": "meta.name", "operator": "==", "value": "name_0"},
                {"field": "meta.name", "operator": "==", "value": "name_1"},
            ],
        },
        '(name contains "name_0" or name contains "name_1")',
    ),
    # NOT operator
    (
        {
            "operator": "NOT",
            "conditions": [
                {"field": "meta.number", "operator": "==", "value": 100},
                {"field": "meta.name", "operator": "==", "value": "name_0"},
            ],
        },
        '!(number = 100 and name contains "name_0")',
    ),
]


@pytest.mark.parametrize("filters, expected", filters_data)
def test_normalize_filters(filters, expected):
    result = normalize_filters(filters)
    assert result == expected


class TestSimpleComparisons:
    """Test individual comparison operators without logical wrappers."""

    def test_equal_string(self):
        filters = {"field": "meta.name", "operator": "==", "value": "test"}
        assert normalize_filters(filters) == 'name contains "test"'

    def test_equal_integer(self):
        filters = {"field": "meta.age", "operator": "==", "value": 30}
        assert normalize_filters(filters) == "age = 30"

    def test_equal_float(self):
        filters = {"field": "meta.score", "operator": "==", "value": 0.95}
        assert normalize_filters(filters) == "score = 0.95"

    def test_equal_bool_true(self):
        filters = {"field": "meta.active", "operator": "==", "value": True}
        assert normalize_filters(filters) == "active = true"

    def test_equal_bool_false(self):
        filters = {"field": "meta.active", "operator": "==", "value": False}
        assert normalize_filters(filters) == "active = false"

    def test_not_equal_string(self):
        filters = {"field": "meta.name", "operator": "!=", "value": "test"}
        assert normalize_filters(filters) == '!(name contains "test")'

    def test_not_equal_integer(self):
        filters = {"field": "meta.age", "operator": "!=", "value": 30}
        assert normalize_filters(filters) == "!(age = 30)"

    def test_greater_than(self):
        filters = {"field": "meta.age", "operator": ">", "value": 30}
        assert normalize_filters(filters) == "age > 30"

    def test_greater_than_equal(self):
        filters = {"field": "meta.age", "operator": ">=", "value": 30}
        assert normalize_filters(filters) == "age >= 30"

    def test_less_than(self):
        filters = {"field": "meta.age", "operator": "<", "value": 30}
        assert normalize_filters(filters) == "age < 30"

    def test_less_than_equal(self):
        filters = {"field": "meta.age", "operator": "<=", "value": 30}
        assert normalize_filters(filters) == "age <= 30"

    def test_range_with_string_date(self):
        filters = {"field": "meta.date", "operator": ">=", "value": "2023-01-01"}
        assert normalize_filters(filters) == 'date >= "2023-01-01"'

    def test_in_strings(self):
        filters = {"field": "meta.status", "operator": "in", "value": ["open", "closed"]}
        assert normalize_filters(filters) == "status in ('open', 'closed')"

    def test_in_integers(self):
        filters = {"field": "meta.code", "operator": "in", "value": [1, 2, 3]}
        assert normalize_filters(filters) == "code in (1, 2, 3)"

    def test_in_floats(self):
        filters = {"field": "meta.score", "operator": "in", "value": [0.5, 1.5]}
        assert normalize_filters(filters) == "score in (0.5, 1.5)"

    def test_not_in(self):
        filters = {"field": "meta.status", "operator": "not in", "value": ["draft", "archived"]}
        assert normalize_filters(filters) == "!(status in ('draft', 'archived'))"


class TestMetaPrefixStripping:
    """Test that 'meta.' prefix is correctly stripped from field names."""

    def test_meta_prefix_stripped(self):
        filters = {"field": "meta.category", "operator": "==", "value": "tech"}
        assert normalize_filters(filters) == 'category contains "tech"'

    def test_no_meta_prefix_unchanged(self):
        filters = {"field": "category", "operator": "==", "value": "tech"}
        assert normalize_filters(filters) == 'category contains "tech"'


class TestStringEscaping:
    """Test proper escaping of special characters in YQL strings."""

    def test_escape_double_quotes(self):
        filters = {"field": "meta.title", "operator": "==", "value": 'say "hello"'}
        assert normalize_filters(filters) == 'title contains "say \\"hello\\""'

    def test_escape_backslash(self):
        filters = {"field": "meta.path", "operator": "==", "value": "C:\\Users\\test"}
        assert normalize_filters(filters) == 'path contains "C:\\\\Users\\\\test"'

    def test_escape_in_list(self):
        filters = {"field": "meta.tag", "operator": "in", "value": ["it's", "a \"test\""]}
        result = normalize_filters(filters)
        # in() uses single quotes, so single quotes are escaped but double quotes are left as-is
        assert result == "tag in ('it\\'s', 'a \"test\"')"


class TestErrorCases:
    """Test that appropriate errors are raised for invalid inputs."""

    def test_filters_not_dict(self):
        with pytest.raises(FilterError):
            normalize_filters("invalid")

    def test_invalid_logical_operator(self):
        with pytest.raises(FilterError):
            normalize_filters({"operator": "INVALID", "conditions": []})

    def test_missing_operator_in_logical(self):
        with pytest.raises(FilterError):
            normalize_filters({"conditions": []})

    def test_missing_conditions_in_logical(self):
        with pytest.raises(FilterError):
            normalize_filters({"operator": "AND"})

    def test_missing_operator_in_comparison(self):
        with pytest.raises(FilterError):
            normalize_filters({"field": "meta.type", "value": "article"})

    def test_missing_value_in_comparison(self):
        with pytest.raises(FilterError):
            normalize_filters({"field": "meta.type", "operator": "=="})

    def test_unknown_comparison_operator(self):
        with pytest.raises(FilterError):
            normalize_filters({"field": "meta.type", "operator": "~=", "value": "article"})

    def test_equal_none_raises_error(self):
        with pytest.raises(FilterError, match="None/null"):
            normalize_filters({"field": "meta.type", "operator": "==", "value": None})

    def test_not_equal_none_raises_error(self):
        with pytest.raises(FilterError, match="None/null"):
            normalize_filters({"field": "meta.type", "operator": "!=", "value": None})

    def test_greater_than_none_raises_error(self):
        with pytest.raises(FilterError, match="None"):
            normalize_filters({"field": "meta.age", "operator": ">", "value": None})

    def test_range_with_list_raises_error(self):
        with pytest.raises(FilterError):
            normalize_filters({"field": "meta.age", "operator": ">", "value": [1, 2]})

    def test_in_with_non_list_raises_error(self):
        with pytest.raises(FilterError):
            normalize_filters({"field": "meta.status", "operator": "in", "value": "open"})

    def test_not_in_with_non_list_raises_error(self):
        with pytest.raises(FilterError):
            normalize_filters({"field": "meta.status", "operator": "not in", "value": "open"})

    def test_in_with_unsupported_type_raises_error(self):
        with pytest.raises(FilterError, match="Unsupported value type"):
            normalize_filters({"field": "meta.data", "operator": "in", "value": [{"nested": True}]})

    def test_unsupported_value_type_raises_error(self):
        with pytest.raises(FilterError, match="Unsupported filter value type"):
            normalize_filters({"field": "meta.data", "operator": ">", "value": {"nested": True}})


class TestEdgeCases:
    """Test edge cases that could produce invalid YQL."""

    def test_bool_in_list(self):
        """Bool is a subclass of int in Python. str(True) gives 'True' not 'true'."""
        filters = {"field": "meta.active", "operator": "in", "value": [True, False]}
        result = normalize_filters(filters)
        assert result == "active in (true, false)"

    def test_empty_list_in(self):
        """Empty list in 'in' operator would produce 'field in ()' — should raise error."""
        with pytest.raises(FilterError):
            normalize_filters({"field": "meta.status", "operator": "in", "value": []})

    def test_empty_list_not_in(self):
        """Empty list in 'not in' operator — should raise error."""
        with pytest.raises(FilterError):
            normalize_filters({"field": "meta.status", "operator": "not in", "value": []})

    def test_empty_conditions_and(self):
        """Empty conditions list in AND would produce '()' — should raise error."""
        with pytest.raises(FilterError):
            normalize_filters({"operator": "AND", "conditions": []})

    def test_empty_conditions_or(self):
        """Empty conditions list in OR would produce '()' — should raise error."""
        with pytest.raises(FilterError):
            normalize_filters({"operator": "OR", "conditions": []})

    def test_float_inf(self):
        """float('inf') would produce 'inf' string — should raise error."""
        with pytest.raises(FilterError):
            normalize_filters({"field": "meta.val", "operator": ">", "value": float("inf")})

    def test_float_nan(self):
        """float('nan') would produce 'nan' string — should raise error."""
        with pytest.raises(FilterError):
            normalize_filters({"field": "meta.val", "operator": "==", "value": float("nan")})

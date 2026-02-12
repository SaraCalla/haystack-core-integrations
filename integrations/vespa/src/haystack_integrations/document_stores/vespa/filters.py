# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

from haystack.errors import FilterError


def normalize_filters(filters: dict[str, Any]) -> str:
    """
    Converts Haystack filters to a Vespa YQL WHERE clause string.

    Returns a string that can be appended to a YQL ``SELECT ... WHERE ...`` statement.
    For example: ``type contains "article" and date >= "2015-01-01"``

    :param filters: A dictionary of Haystack filters in the universal filter format.
    :returns: A YQL WHERE clause string.
    """
    if not isinstance(filters, dict):
        msg = "Filters must be a dictionary"
        raise FilterError(msg)

    if "field" in filters:
        return _parse_comparison_condition(filters)
    return _parse_logical_condition(filters)


def _parse_logical_condition(condition: dict[str, Any]) -> str:
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in {condition}"
        raise FilterError(msg)

    operator = condition["operator"]
    if not condition["conditions"]:
        msg = f"'{operator}' operator requires at least one condition"
        raise FilterError(msg)
    conditions = [_parse_comparison_condition(c) for c in condition["conditions"]]

    if operator == "AND":
        return "(" + " and ".join(conditions) + ")"
    elif operator == "OR":
        return "(" + " or ".join(conditions) + ")"
    elif operator == "NOT":
        inner = " and ".join(conditions)
        return f"!({inner})"
    else:
        msg = f"Unknown logical operator '{operator}'"
        raise FilterError(msg)


def _parse_comparison_condition(condition: dict[str, Any]) -> str:
    if "field" not in condition:
        # 'field' key is only found in comparison dictionaries.
        # We assume this is a logic dictionary since it's not present.
        return _parse_logical_condition(condition)

    field: str = condition["field"]

    if field.startswith("meta."):
        # Remove the "meta." prefix if present.
        # Documents are flattened when stored in Vespa
        # so we don't need to specify the "meta." prefix.
        field = field[5:]

    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "value" not in condition:
        msg = f"'value' key missing in {condition}"
        raise FilterError(msg)

    operator: str = condition["operator"]
    value: Any = condition["value"]

    if operator not in COMPARISON_OPERATORS:
        msg = f"Unknown comparison operator '{operator}'"
        raise FilterError(msg)

    return COMPARISON_OPERATORS[operator](field, value)


def _escape_yql_string(value: str) -> str:
    """Escape a string value for use in YQL."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _format_value(value: Any) -> str:
    """Format a value for YQL."""
    if isinstance(value, str):
        return f'"{_escape_yql_string(value)}"'
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
            msg = f"Filter value {value} is not supported in Vespa YQL"
            raise FilterError(msg)
        return str(value)
    else:
        msg = f"Unsupported filter value type: {type(value)}"
        raise FilterError(msg)


def _equal(field: str, value: Any) -> str:
    if value is None:
        # TODO: check if it's possible to filter for None/null values in Vespa maybe mapping
        msg = "Filtering for None/null values is not supported in Vespa"
        raise FilterError(msg)
    if isinstance(value, str):
        return f'{field} contains "{_escape_yql_string(value)}"'
    return f"{field} = {_format_value(value)}"


def _not_equal(field: str, value: Any) -> str:
    if value is None:
        msg = "Filtering for None/null values is not supported in Vespa"
        raise FilterError(msg)
    if isinstance(value, str):
        return f'!({field} contains "{_escape_yql_string(value)}")'
    return f"!({field} = {_format_value(value)})"


def _greater_than(field: str, value: Any) -> str:
    _validate_range_value(value)
    return f"{field} > {_format_value(value)}"


def _greater_than_equal(field: str, value: Any) -> str:
    _validate_range_value(value)
    return f"{field} >= {_format_value(value)}"


def _less_than(field: str, value: Any) -> str:
    _validate_range_value(value)
    return f"{field} < {_format_value(value)}"


def _less_than_equal(field: str, value: Any) -> str:
    _validate_range_value(value)
    return f"{field} <= {_format_value(value)}"


def _in(field: str, value: Any) -> str:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'in' or 'not in' comparators"
        raise FilterError(msg)
    if not value:
        msg = f"'{field}' filter with 'in' or 'not in' requires a non-empty list"
        raise FilterError(msg)
    formatted_values = []
    for v in value:
        if isinstance(v, str):
            # in() uses single quotes, so escape backslashes and single quotes
            escaped = v.replace("\\", "\\\\").replace("'", "\\'")
            formatted_values.append(f"'{escaped}'")
        elif isinstance(v, bool):
            # Check bool before int since bool is a subclass of int in Python
            formatted_values.append("true" if v else "false")
        elif isinstance(v, (int, float)):
            formatted_values.append(str(v))
        else:
            msg = f"Unsupported value type in 'in' list: {type(v)}"
            raise FilterError(msg)
    return f"{field} in ({', '.join(formatted_values)})"


def _not_in(field: str, value: Any) -> str:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'in' or 'not in' comparators"
        raise FilterError(msg)
    return f"!({_in(field, value)})"


def _validate_range_value(value: Any) -> None:
    if value is None:
        msg = "Range operators do not support None values"
        raise FilterError(msg)
    if isinstance(value, list):
        msg = f"Filter value can't be of type {type(value)} using range operators"
        raise FilterError(msg)


COMPARISON_OPERATORS = {
    "==": _equal,
    "!=": _not_equal,
    ">": _greater_than,
    ">=": _greater_than_equal,
    "<": _less_than,
    "<=": _less_than_equal,
    "in": _in,
    "not in": _not_in,
}

import pytest
import json
from rust_python_lib import (
    process_numbers,
    concatenate_strings,
    create_person,
    analyze_data,
    process_mixed_data,
    fibonacci,
    Person,
    DataPoint,
    ProcessResult,
)


class TestProcessNumbers:
    def test_basic(self):
        result = process_numbers([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result.sum == 15.0
        assert result.average == 3.0
        assert result.min == 1.0
        assert result.max == 5.0
        assert result.count == 5

    def test_empty_list(self):
        with pytest.raises(ValueError):
            process_numbers([])

    def test_single_value(self):
        result = process_numbers([42.0])
        assert result.sum == 42.0
        assert result.average == 42.0
        assert result.min == 42.0
        assert result.max == 42.0


class TestConcatenateStrings:
    def test_basic(self):
        result = concatenate_strings(["a", "b", "c"], None)
        assert result == "a, b, c"

    def test_custom_separator(self):
        result = concatenate_strings(["a", "b", "c"], "|")
        assert result == "a|b|c"

    def test_empty_list(self):
        result = concatenate_strings([], None)
        assert result == ""


class TestPerson:
    def test_creation(self):
        person = Person("Alice", 30, "alice@example.com")
        assert person.name == "Alice"
        assert person.age == 30
        assert person.email == "alice@example.com"

    def test_is_adult(self):
        adult = Person("Adult", 18, None)
        child = Person("Child", 17, None)
        assert adult.is_adult() is True
        assert child.is_adult() is False

    def test_json_roundtrip(self):
        original = Person("Bob", 25, "bob@example.com")
        json_str = original.to_json()
        restored = Person.from_json(json_str)
        assert original.name == restored.name
        assert original.age == restored.age
        assert original.email == restored.email

    def test_greet(self):
        person = Person("Charlie", 40, None)
        greeting = person.greet()
        assert "Charlie" in greeting
        assert "40" in greeting


class TestDataPoint:
    def test_distance(self):
        point = DataPoint(3.0, 4.0, "test")
        assert point.distance_from_origin() == 5.0

    def test_scale(self):
        point = DataPoint(2.0, 3.0, "test")
        point.scale(2.0)
        assert point.x == 4.0
        assert point.y == 6.0


class TestAnalyzeData:
    def test_basic(self):
        points = [
            DataPoint(1.0, 2.0, "A"),
            DataPoint(3.0, 4.0, "B"),
            DataPoint(5.0, 6.0, "A"),
        ]
        result = analyze_data(points)
        assert result["total_points"] == 3
        assert result["average_x"] == pytest.approx(3.0)
        assert result["average_y"] == pytest.approx(4.0)
        assert result["label_counts"] == {"A": 2, "B": 1}


class TestProcessMixedData:
    def test_basic(self):
        mixed = ["hello", 42, 3.14, True]
        result = process_mixed_data(mixed)
        assert len(result) == 4
        assert "item_0" in result
        assert "str:hello" in result["item_0"]


class TestFibonacci:
    def test_first_10(self):
        result = fibonacci(10)
        expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        assert result == expected

    def test_zero(self):
        assert fibonacci(0) == []

    def test_one(self):
        assert fibonacci(1) == [0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

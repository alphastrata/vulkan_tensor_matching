#!/usr/bin/env python3
"""
Example usage of the Rust-Python library.
"""

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
    VERSION,
    AUTHOR,
)


def main():
    print(f"Using library version: {VERSION}")
    print(f"Author: {AUTHOR}")
    print("-" * 50)

    # Example 1: Process numbers
    print("1. Processing numbers:")
    numbers = [1.5, 2.5, 3.5, 4.5, 5.5]
    result: ProcessResult = process_numbers(numbers)
    print(f"   Numbers: {numbers}")
    print(f"   Sum: {result.sum}, Average: {result.average}")
    print(f"   Min: {result.min}, Max: {result.max}, Count: {result.count}")
    print()

    # Example 2: String concatenation
    print("2. String concatenation:")
    strings = ["Hello", "World", "from", "Rust"]
    concatenated = concatenate_strings(strings, separator=" | ")
    print(f"   Strings: {strings}")
    print(f"   Result: {concatenated}")
    print()

    # Example 3: Working with Person struct
    print("3. Working with Person:")
    person = create_person("Alice", 30, "alice@example.com")
    print(f"   Created: {person}")
    print(f"   Is adult: {person.is_adult()}")
    print(f"   Greeting: {person.greet()}")

    # JSON serialization/deserialization
    json_str = person.to_json()
    print(f"   JSON: {json_str}")

    person_copy = Person.from_json(json_str)
    print(f"   From JSON: {person_copy}")
    print()

    # Example 4: DataPoint analysis
    print("4. DataPoint analysis:")
    points = [
        DataPoint(1.0, 2.0, "A"),
        DataPoint(3.0, 4.0, "B"),
        DataPoint(5.0, 6.0, "A"),
        DataPoint(7.0, 8.0, "C"),
        DataPoint(2.0, 3.0, "B"),
    ]

    print(f"   Points: {len(points)} data points")
    for i, point in enumerate(points):
        print(
            f"     Point {i}: ({point.x}, {point.y}) '{point.label}' "
            f"distance={point.distance_from_origin():.2f}"
        )

    analysis = analyze_data(points)
    print("   Analysis:")
    for key, value in analysis.items():
        print(f"     {key}: {value}")
    print()

    # Example 5: Process mixed data
    print("5. Processing mixed data:")
    mixed_list = ["hello", 42, 3.14, True, False, "world", 100]
    mixed_result = process_mixed_data(mixed_list)
    print(f"   Input: {mixed_list}")
    print(f"   Result: {json.dumps(mixed_result, indent=4)}")
    print()

    # Example 6: Fibonacci sequence
    print("6. Fibonacci sequence:")
    for n in [5, 10, 15]:
        fib_seq = fibonacci(n)
        print(f"   First {n} Fibonacci numbers: {fib_seq}")

    # Example 7: Error handling
    print("\n7. Error handling examples:")
    try:
        process_numbers([])
    except ValueError as e:
        print(f"   Expected error: {e}")

    try:
        create_person("", 30, None)
    except ValueError as e:
        print(f"   Expected error: {e}")

    print("\n" + "=" * 50)
    print("All examples completed successfully!")


if __name__ == "__main__":
    main()

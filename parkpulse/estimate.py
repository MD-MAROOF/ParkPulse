"""Parking capacity estimation helpers."""


def estimate_spots_from_cars(n_cars: int, occupancy: float = 0.6) -> int:
    """Estimate total parking spots from observed car count and occupancy.

    Formula: spots = ceil(n_cars / occupancy).

    Args:
        n_cars: Number of cars detected (or observed).
        occupancy: Assumed occupancy ratio in [0, 1] (default 0.6).

    Returns:
        Estimated total number of parking spots (at least n_cars).
    """
    if occupancy <= 0 or occupancy > 1:
        raise ValueError("occupancy must be in (0, 1]")
    if n_cars < 0:
        raise ValueError("n_cars must be non-negative")
    import math
    return max(n_cars, math.ceil(n_cars / occupancy))


def estimate_spots_from_area(area_m2: float, m2_per_spot: float = 30.0) -> int:
    """Estimate number of parking spots from paved parking area.

    Formula: spots = floor(area_m2 / m2_per_spot).

    Args:
        area_m2: Parking area in square meters.
        m2_per_spot: Assumed area per spot in m² (default 30.0).

    Returns:
        Estimated number of parking spots (non-negative integer).
    """
    if m2_per_spot <= 0:
        raise ValueError("m2_per_spot must be positive")
    if area_m2 < 0:
        raise ValueError("area_m2 must be non-negative")
    import math
    return max(0, int(area_m2 // m2_per_spot))


def combine_estimates(a: int, b: int) -> int:
    """Combine two spot estimates by simple average, rounded to nearest int.

    Args:
        a: First estimate (number of spots).
        b: Second estimate (number of spots).

    Returns:
        Round(average of a and b); non-negative integer.
    """
    return max(0, round((a + b) / 2))

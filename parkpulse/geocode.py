"""Geocoding utilities using Nominatim."""

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError


def geocode_place(place: str) -> tuple[float, float]:
    """Geocode a place name or address to (latitude, longitude) using Nominatim.

    Args:
        place: Place name, address, or search string to geocode.

    Returns:
        A tuple (latitude, longitude) in decimal degrees (WGS84).

    Raises:
        geopy.exc.GeocoderTimedOut: If the request times out.
        geopy.exc.GeocoderServiceError: If the geocoding service fails.
        AttributeError: If no result is found for the query.
    """
    geolocator = Nominatim(
        user_agent="ParkPulse/1.0 (parking availability and capacity estimation; https://github.com/parkpulse)"
    )
    location = geolocator.geocode(place)
    if location is None:
        raise ValueError(f"No result found for place: {place!r}")
    return (location.latitude, location.longitude)

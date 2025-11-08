"""
Simple test to check MeteoGalicia API access
"""
import requests
import json

print("Testing MeteoGalicia API access...")
print("=" * 60)

# Test 1: Get list of stations
STATIONS_URL = "https://servizos.meteogalicia.gal/mgrss/observacion/listaEstacionsMeteo.action"
print("\nTest 1: Fetching stations list...")
print(f"URL: {STATIONS_URL}")

try:
    response = requests.get(STATIONS_URL, timeout=30)
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")

    if response.status_code == 200:
        data = response.json()
        stations_count = len(data.get('listaEstacionsMeteo', []))
        print(f"✓ SUCCESS: Retrieved {stations_count} stations")
        if stations_count > 0:
            print(f"\nExample station:")
            print(json.dumps(data['listaEstacionsMeteo'][0], indent=2))
    elif response.status_code == 503:
        print("✗ ERROR 503: Service Unavailable")
        print("This is the error you mentioned!")
    else:
        print(f"✗ Unexpected status code: {response.status_code}")
        print(f"Response: {response.text[:500]}")

except requests.exceptions.RequestException as e:
    print(f"✗ Request failed: {e}")

print("\n" + "=" * 60)

# Test 2: Get daily data for a station
BASE_URL = "https://servizos.meteogalicia.gal/mgrss/observacion/datosDiariosEstacionsMeteo.action"
print("\nTest 2: Fetching daily data...")
print(f"URL: {BASE_URL}")

try:
    # Simple query without parameters to see what happens
    response = requests.get(BASE_URL, timeout=30)
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"✓ SUCCESS: API responded")
        print(f"Response keys: {list(data.keys())}")
    elif response.status_code == 503:
        print("✗ ERROR 503: Service Unavailable")
        print("This is the error you mentioned!")
    else:
        print(f"✗ Unexpected status code: {response.status_code}")

except requests.exceptions.RequestException as e:
    print(f"✗ Request failed: {e}")

print("\n" + "=" * 60)
print("Test complete!")

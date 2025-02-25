import camelot
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ----------------------------
# Step 1: Extract and Clean PDF Table Data
# ----------------------------
# Read all tables from the PDF.
tables = camelot.read_pdf("<insert-PDF-here>", pages='all', flavor='stream')
df_list = [table.df for table in tables]
df = pd.concat(df_list, ignore_index=True)

# Find the header row containing "Brewery Name"
header_idx = None
for i, row in df.iterrows():
    if row.astype(str).str.contains("Brewery Name", case=False).any():
        header_idx = i
        break

if header_idx is None:
    raise ValueError("Could not find a header row with 'Brewery Name' in the extracted data.")

# Set the header and remove rows above it.
df.columns = df.iloc[header_idx]
df = df.iloc[header_idx+1:].reset_index(drop=True)
df = df[df[df.columns[0]].notnull()].reset_index(drop=True)

# Save the intermediate CSV (optional)
df.to_csv("breweries.csv", index=False)
print("PDF converted to CSV and saved as 'breweries.csv'.")

# ----------------------------
# Step 2: Geocoding
# ----------------------------
# Ensure the expected columns exist.
expected_cols = ['Brewery Name', 'Address', 'City', 'State', 'Zip']
for col in expected_cols:
    if col not in df.columns:
        raise KeyError(f"Expected column '{col}' not found. Found columns: {df.columns.tolist()}")

# Combine address components into a full address.
df['full_address'] = df['Address'] + ", " + df['City'] + ", " + df['State'] + " " + df['Zip'].astype(str)

# Initialize geolocator with a longer timeout.
geolocator = Nominatim(user_agent="brewery_cluster", timeout=10)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# Geocoding functions with progress logging.
def get_lat(address):
    try:
        location = geocode(address)
        if location:
            print(f"Geocoded: {address} => {location.latitude}")
        else:
            print(f"Geocoding failed: {address}")
        return location.latitude if location else None
    except Exception as e:
        print(f"Error geocoding {address}: {e}")
        return None

def get_lon(address):
    try:
        location = geocode(address)
        if location:
            print(f"Geocoded: {address} => {location.longitude}")
        else:
            print(f"Geocoding failed: {address}")
        return location.longitude if location else None
    except Exception as e:
        print(f"Error geocoding {address}: {e}")
        return None

# Apply geocoding to all addresses.
df['latitude'] = df['full_address'].apply(get_lat)
df['longitude'] = df['full_address'].apply(get_lon)

# ----------------------------
# Step 3: Separate Successful and Failed Geocodes
# ----------------------------
df_geo = df[df['latitude'].notnull() & df['longitude'].notnull()].copy()
df_fail = df[df['latitude'].isnull() | df['longitude'].isnull()].copy()

# ----------------------------
# Step 4: Cluster Successfully Geocoded Rows
# ----------------------------
if not df_geo.empty:
    coords = df_geo[['latitude', 'longitude']].to_numpy().astype(float)
    coords_rad = np.radians(coords)
    # 5 miles converted to radians (Earth's radius ≈ 3959 miles)
    epsilon = 5 / 3959.0
    db = DBSCAN(eps=epsilon, min_samples=1, metric='haversine').fit(coords_rad)
    df_geo['cluster'] = db.labels_
else:
    df_geo['cluster'] = pd.Series(dtype=int)

# ----------------------------
# Step 5: Cluster Rows with Failed Geocodes
# ----------------------------
if not df_fail.empty:
    # If available, use 'County' for clustering; otherwise fallback to 'City'
    if 'County' in df_fail.columns:
        counties = df_fail['County'].fillna('Unknown')
        max_cluster = df_geo['cluster'].max() if not df_geo.empty else -1
        unique_counties = counties.unique()
        # Map each unique county to a new cluster number (starting after successful clusters)
        county_to_cluster = {county: i for i, county in enumerate(unique_counties, start=max_cluster+1)}
        df_fail['cluster'] = counties.map(county_to_cluster)
    else:
        cities = df_fail['City'].fillna('Unknown')
        max_cluster = df_geo['cluster'].max() if not df_geo.empty else -1
        unique_cities = cities.unique()
        city_to_cluster = {city: i for i, city in enumerate(unique_cities, start=max_cluster+1)}
        df_fail['cluster'] = cities.map(city_to_cluster)

# ----------------------------
# Step 6: Combine and Save the Final Output
# ----------------------------
df_final = pd.concat([df_geo, df_fail], ignore_index=True)
output_file = "breweries_clustered.csv"
df_final.to_csv(output_file, index=False)
print(f"Final clustered output saved to {output_file}")


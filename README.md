# brewery_cluster
Cluster breweries based on addresses using latitude &amp; longitude.  Those that fail using geo will use counties and cities.  

Steps: 

1. The code digests a PDF file (i.e. that has tables on it of addresses).
2. The code uses geocoding via Nominatim agency.
3. Breweries (aka "Address") are then clustered via numbers.
4. Those that fail geocoding are clustered via county then city.
5. A new .csv file is then created reflecting the clusters.

#########################################
##### Name:Yunyang Ma               #####
##### Uniqname:yuyuma               #####
#########################################
import requests
import re
from collections import Counter
import os
import json
import unittest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
from matplotlib.path import Path
from pathlib import Path as Pathl
random.seed(17)

#step_01
BASE_URL = "https://dsl.richmond.edu/panorama/redlining/static/downloads/geojson/MIDetroit1939.geojson"
resp = requests.get(BASE_URL)
json_str = resp.text
RedliningData= json.loads(json_str)
#print(RedliningData)

#step_02
class DetroitDistrict:
    """
    A class representing a district in Detroit.

    Attributes:
        color_map (dict): A class attribute mapping HOLC grades to colors.
        Coordinates (list): The geographical coordinates defining the district.
        HolcGrade (str): The Home Owners' Loan Corporation grade of the district.
        HolcColor (str): The color associated with the HOLC grade.
        name (str): The name or area number of the district.
        QualitativeDescription (str): A qualitative description of the district.
        RandomLat (float): A randomly chosen latitude within the district (default None).
        RandomLong (float): A randomly chosen longitude within the district (default None).
        MedianIncome (float): The median income of the district (default None).
        CensusTract (str): The census tract of the district (default None).

    Methods:
        set_census_tract: Sets the CensusTract attribute based on RandomLat and RandomLong.

    Args:
        feature (dict): A dictionary containing district data, typically from a geo-spatial dataset.
    """
    color_map = {
        'A': 'darkgreen',
        'B': 'cornflowerblue',
        'C': 'gold',
        'D': 'maroon'
    }
    def __init__(self, feature):
        self.Coordinates = feature['geometry']['coordinates']
        self.HolcGrade = feature['properties']['holc_grade']
        self.HolcColor = self.color_map.get(self.HolcGrade, 'gray')  # Default color if not found
        # Name can be area number
        self.name = feature['properties'].get('name', feature['properties'].get('area_number'))
        self.QualitativeDescription = feature['properties']['area_description_data'].get('8', '')
        self.RandomLat = None
        self.RandomLong = None
        self.MedianIncome = None
        self.CensusTract = None
    def set_census_tract(self):
        self.CensusTract = get_census_tract(self.RandomLat, self.RandomLong)
Districts = [DetroitDistrict(feature) for feature in RedliningData['features']]
#step_03:numpy

fig, ax = plt.subplots()

for district in Districts:
    for polygon_coordinates in district.Coordinates:
        # Ensure the coordinates are in a numpy array format, accounting for potential nested lists
        # Polygon coordinates may be nested in another list if they are MultiPolygons
        if isinstance(polygon_coordinates[0][0], list):  # Check if we have a nested list
            polygon_coordinates = polygon_coordinates[0]
        np_coords = np.array(polygon_coordinates)
        # Create the Polygon patch with the district's color and add it to the plot
        poly = matplotlib.patches.Polygon(np_coords, closed=True, fill=True, edgecolor='black', facecolor=district.HolcColor)
        ax.add_patch(poly)  # Add the polygon patch to the Axes

ax.autoscale()
plt.rcParams["figure.figsize"] = (15,15)
plt.show()

#step_04:latitude and longtitude, comment line by line
ygrid= np.arange(42.1, 42.6, .004)#latitude range
xgrid= np.arange(-83.5, -82.8, .004)#longtitude range
xmesh, ymesh= np.meshgrid(xgrid, ygrid)#mesh gird

points= np.vstack((xmesh.flatten(), ymesh.flatten())).T #Flatten and transpose to get (x, y) points

for j in Districts:
  coordinates = j.Coordinates[0]
  coordinates = coordinates[0]  #Take the nested array
  p = Path(coordinates)  #Create a Path object for the district's polygon
  grid = p.contains_points(points) #Check which points lie inside the district's polygon
  print(j,":", points[random.choice(list(np.where(grid)[0]))]) #Select a random point inside the district
  point= points[random.choice(list(np.where(grid)[0]))]
  j.RandomLat= point[1]
  j.RandomLong= point[0]

#step_05:census tract code
def get_census_tract(lat, lon):
    """
    Retrieves the census tract code for a given latitude and longitude.

    This function makes a request to the FCC's API to find the census block data for the specified coordinates.
    It then extracts and returns the census tract code from the response.

    Args:
        lat (float): The latitude of the location.
        lon (float): The longitude of the location.

    Returns:
        str: The census tract code if found, otherwise None.
    """
    url = f"https://geo.fcc.gov/api/census/block/find?latitude={lat}&longitude={lon}&censusYear=2010&format=json"
    response = requests.get(url)
    data = response.json()
    # Initialize fips_code to None
    fips_code = None

    # Check if 'Block' key exists in the response
    if 'Block' in data:
        # Check if 'FIPS' key is in the Block data
        fips_code = data['Block'].get('FIPS')

    # Check if fips_code is not None and its length is sufficient
    if fips_code and len(fips_code) >= 9:
        tract_code = fips_code[5:-4]
        #print(tract_code)
        return tract_code
    else:
        print("Block data or FIPS code not found, or FIPS code is too short.")
        return None  # Return None if the FIPS code is not found or is too short

#step_06:
def fetch_median_income():
    """
    Fetches and updates the median household income data for each district.

    Returns:
        None: This function does not return a value. It modifies the `Districts` list in place.
    """
    # This base_url will be for the ACS5 2018 dataset
    base_url = "https://api.census.gov/data/2018/acs/acs5?get=B19013_001E&for=tract:*&in=state:26&key=37acc6295e2a7183cb8eee394c386ec0fb095a33"
    # B19013_001E is a common variable for Median Household Income in the past 12 months

    response = requests.get(base_url)

    if response.status_code == 200:
        try:
            data = response.json()
            # Skip the header row in the data
            for row in data[1:]:
                income = row[0]  # The median income value
                state = row[1]  # The state code
                county = row[2]  # The county code
                tract = row[3]  # The census tract code

                # Now you need to find the district object that this tract corresponds to
                # and update the median income for that district.
                for i in range (0,len(Districts)):
                    if Districts[i].CensusTract == tract:
                        Districts[i].MedianIncome = income
                        break

        except json.JSONDecodeError:
            print("Error decoding JSON:", response.text)

    else:
        print("Failed to retrieve data:", response.status_code)

#step_07: cache
CACHE_FILENAME = "cache.json"
def open_cache():
    ''' opens the cache file if it exists and loads the JSON into
    a dictionary, which it then returns.
    if the cache file doesn't exist, creates a new cache dictionary
    Parameters
    ----------
    None
    Returns
    -------
    The opened cache
    '''
    try:
        cache_file = open(CACHE_FILENAME, 'r')
        cache_contents = cache_file.read()
        cache_dict = json.loads(cache_contents)
        cache_file.close()
    except:
        cache_dict = {}
    return cache_dict

def save_cache(cache_dict):
    ''' saves the current state of the cache to disk
    Parameters
    ----------
    cache_dict: dict
        The dictionary to save
    Returns
    -------
    None
    '''
    dumped_json_cache = json.dumps(cache_dict)
    fw = open(CACHE_FILENAME,"w")
    fw.write(dumped_json_cache)
    fw.close()

def district_with_cache(dictionary,lat_long):
    """
    Retrieves or updates district data using caching to improve efficiency.

    Args:
        dictionary (dict): The cache dictionary storing district data.
        lat_long (list): A list containing the latitude and longitude of the district.

    Returns:
        dict or None: Returns the cached data for the district if a cache miss occurs and new data is fetched.
                      Returns None if the data is found in the cache.
    """
    key= str(lat_long)
    if key in dictionary:
        for district in Districts:
            if [district.RandomLat, district.RandomLong] == lat_long:
                district.MedianIncome = dictionary[key]['income_info']
                district.CensusTract = dictionary[key]['census tract']
                #print('Cache hit for district:', district)
                break
    else:
        for district in Districts:
            if [district.RandomLat, district.RandomLong] == lat_long:
                district.CensusTract = get_census_tract(district.RandomLat, district.RandomLong)
                fetch_median_income()  # Assuming this updates all districts' MedianIncome
                dictionary[key] = {
                    'income_info': district.MedianIncome,
                    'random_lat&Long': (district.RandomLat, district.RandomLong),
                    'census tract': district.CensusTract
                }
                save_cache(dictionary)
                #print('Cache miss, data retrieved and saved for district:', district)
                break
        return dictionary[key]

DISTRICT_CACHE= open_cache()
for district in Districts:
    district_with_cache(DISTRICT_CACHE, [district.RandomLat, district.RandomLong])

#step_08:assign income
def mean_median(grade):
    """
    Calculates and prints the mean and median income for districts of a specific HOLC grade.

    Args:
        grade (str): The HOLC grade (e.g., 'A', 'B', 'C', 'D') to filter the districts by.

    Prints:
        The mean and median income for the specified grade.
    """
    income = [district.MedianIncome for district in Districts if district.HolcGrade == grade if district.MedianIncome!= None]
    int_income = [int(item) for item in income]

    mean_income = np.mean(int_income) if int_income else None
    median_income = np.median(int_income) if int_income else None

    print(f"{grade} Grade Mean Income: {mean_income}")
    print(f"{grade} Grade Median Income: {median_income}\n")

mean_median('A')
mean_median('B')
mean_median('C')
mean_median('D')

#step_09:common words
descriptions = {
    'A': str([district.QualitativeDescription for district in Districts if district.HolcGrade == 'A']),
    'B': str([district.QualitativeDescription for district in Districts if district.HolcGrade == 'B']),
    'C': str([district.QualitativeDescription for district in Districts if district.HolcGrade == 'C']),
    'D': str([district.QualitativeDescription for district in Districts if district.HolcGrade == 'D'])
}
# Common filler words to exclude
filler_words = set(['the', 'of', 'and', 'in', 'to', 'with', 'some', 'for', 'might',
                     'have', 'has', 'its', 'own','on','should','is',
                     '1st', '2nd','3rd','4th','b'])

# Function to clean and split text into words, excluding filler words
def clean_and_split(text):
    """
    Splits a text string into words and removes specified filler words.

    Args:
        text (str): The text string to be split and cleaned.

    Returns:
        list: A list of words from the text, with filler words removed.

    Raises:
        ValueError: If the input is not a string.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    words = re.findall(r'\b\w+\b', text.lower())  # Split into words and convert to lowercase
    return [word for word in words if word not in filler_words]

# Split each description into words, excluding filler words
split_descriptions = {grade: clean_and_split(text) for grade, text in descriptions.items()}

# Count the frequency of each word in each grade
word_counts = {grade: Counter(words) for grade, words in split_descriptions.items()}

# Find words that are unique to each category by subtracting counts of other grades
unique_word_counts = {
    grade: counts - sum((word_counts[other_grade] for other_grade in word_counts if other_grade != grade), Counter())
    for grade, counts in word_counts.items()
}

# Find the 10 most common unique words for each grade
A_10_Most_Common = [word for word, count in unique_word_counts['A'].most_common(10)]
B_10_Most_Common = [word for word, count in unique_word_counts['B'].most_common(10)]
C_10_Most_Common = [word for word, count in unique_word_counts['C'].most_common(10)]
D_10_Most_Common = [word for word, count in unique_word_counts['D'].most_common(10)]

print("A - 10 Most Common Unique Words:", A_10_Most_Common)
print("B - 10 Most Common Unique Words:", B_10_Most_Common)
print("C - 10 Most Common Unique Words:", C_10_Most_Common)
print("D - 10 Most Common Unique Words:", D_10_Most_Common)




import builtins
import main

# The inputs we want to provide to main.py interactive CLI:
# 1) VIIRS: '1'
# 2) Custom BBox: '3'
# 3) Min Lat: '39.0'
# 4) Max Lat: '42.0'
# 5) Min Lon: '-124.0'
# 6) Max Lon: '-120.0'
# 7) Sentinel-2: '1'
# 8) Start date: '01-08-2021'
# 9) End date: '31-08-2021'
# 10) Target images: '1'

inputs = [
    '1',            # VIIRS
    '3',            # Custom
    '39.0',         # Min Lat
    '42.0',         # Max Lat
    '-124.0',       # Min Lon
    '-120.0',       # Max Lon
    '1',            # Sentinel-2
    '01-08-2021',   # Start date
    '31-08-2021',   # End date
    '1'            # Target images
]

input_idx = 0

def mock_input(prompt=""):
    global input_idx
    if input_idx < len(inputs):
        val = inputs[input_idx]
        input_idx += 1
        return val
    return ""

builtins.input = mock_input

# Run the original main
if __name__ == "__main__":
    main.main()

import os
pb_path = r'C:\Users\Aaron\.gemini\antigravity-ide\conversations\a38ffe04-ddca-4014-a740-356fe8c784f0.pb'
try:
    with open(pb_path, 'rb') as f:
        data = f.read(32)
        print("Magic bytes (hex):", data.hex())
except Exception as e:
    print(f"Error: {e}")

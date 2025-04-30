# Databricks notebook source
import glob
from PIL import Image
import matplotlib.pyplot as plt

# COMMAND ----------

uc_output_volume = "/Volumes/kyra_wulffert/default/petrology_augmented"   

paths = glob.glob(f"{uc_output_volume}/*.png")[:100]
n = len(paths)
if n == 0:
    raise FileNotFoundError(f"No PNGs found in {uc_output_volume}")

cols = 10
rows = (n + cols - 1) // cols

plt.figure(figsize=(cols*1.5, rows*1.5))   # ~1.5" per image
for idx, p in enumerate(paths):
    img = Image.open(p)
    ax = plt.subplot(rows, cols, idx+1)
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()
plt.savefig(f"{uc_output_volume}/petrology_grid.png", format="png", dpi=150, bbox_inches="tight")

# COMMAND ----------

img_path = f"{uc_output_volume}/petrology_grid.png"

img = Image.open(img_path)

plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.axis('off')
plt.show()

# COMMAND ----------

import base64
import io
from PIL import Image

# 1) Load your grid from the UC volume
grid = Image.open(img_path)

# 2) Base64‐encode it
buf = io.BytesIO()
grid.save(buf, format="PNG", optimize=True)
b64 = base64.b64encode(buf.getvalue()).decode()

# 3) Pass the data URI to OpenSeadragon
html = f"""
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.0.0/openseadragon.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.0.0/openseadragon.min.js"></script>

<div id="osd1" style="width:800px; height:800px;"></div>
<script>
  OpenSeadragon({{
    id: "osd1",
    prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.0.0/images/",
    tileSources: {{
      type: "image",
      url: "data:image/png;base64,{b64}"
    }},
    defaultZoomLevel: 1,
    gestureSettingsMouse: {{ scrollToZoom: true, clickToZoom: true }}
  }});
</script>
"""
displayHTML(html)
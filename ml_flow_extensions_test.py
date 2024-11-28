# Databricks notebook source
from PIL import Image
import IPython.display as display
import io
import base64

# Define the Unity Catalog image path
unity_catalog_image_path = "/Volumes/kyra_wulffert/default/images/pid.png"

image = Image.open(unity_catalog_image_path)
jpeg_image = io.BytesIO()
image.convert("RGB").save(jpeg_image, format="JPEG")  # Convert to RGB (removes transparency)
jpeg_image.seek(0)
jpeg_converted_image = Image.open(jpeg_image)


encoded_image = base64.b64encode(jpeg_image.read()).decode("utf-8")

display.display(jpeg_converted_image)

# COMMAND ----------

endpoint_name = "Qwen-vl-72b"

from openai import OpenAI
from mlflow.utils.databricks_utils import get_databricks_host_creds
# from mlflow_extensions.databricks.prebuilt import prebuilt
from pydantic import BaseModel
import typing as t

workspace_host = spark.conf.get("spark.databricks.workspaceUrl")
# endpoint_name = f"https://{workspace_host}/serving-endpoints/{endpoint_name}/invocations"
endpoint_name = f"https://{workspace_host}/serving-endpoints"
token = get_databricks_host_creds().token

client = OpenAI(
  base_url=endpoint_name,
  api_key=token
)

# COMMAND ----------

import glob

# volume_folder = "/Volumes/kyra_wulffert/vision/images"
volume_folder = "/Volumes/kyra_wulffert/default/images"
directory = glob.glob(volume_folder + '/*.*')

# COMMAND ----------

import base64
import requests

def extract_from_image(image_path, prompt):
    """
    Process a single image: encode to Base64, call the API, and return the result.
    
    Args:
        image_path (str): Path to the image.
    
    Returns:
        dict: Dictionary containing the image path and the API response.
    """
    try:
        # Encode the image to Base64
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        # Call the API
        response = client.chat.completions.create(
            model="Qwen-vl-72b",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                ],
            }],
            max_tokens=256,
        )

        # Extract the response content
        return {
            "image_path": image_path,
            "response": response.choices[0].message.content  
        }

    except Exception as e:
        # Return error details for troubleshooting
        print(f"Error processing image {image_path}: {e}")
        return {
            "image_path": image_path,
            "error": str(e)
        }


# COMMAND ----------

prompt = "What’s in this image?"

# COMMAND ----------

# Process the directory of images
data = [extract_from_image(x, prompt) for x in directory]

# Convert to Spark DataFrame
if data:
    results_df = spark.createDataFrame(data)
    results_df.show(truncate=False)
else:
    print("No valid data to process.")

# COMMAND ----------

prompt_instruction = """
* Task Definition:
    "Analyze the provided P&ID diagram. Identify all unique component IDs, tag numbers, or any identifying labels associated with the symbols in the diagram."

* Clarification of IDs:
    "IDs typically include alphanumeric codes, such as 'P-101' for pumps, 'V-202' for valves, or similar labels. Extract and list these IDs."

* Association with Components:
    "For each extracted ID, if possible, associate it with the corresponding component type (e.g., pump, valve, pipe). If no association is clear, specify 'unknown'."

* Output Format:
    "Present your findings as a structured list:
    ID: [Extracted ID]
    Component Type: [Type or 'unknown']"

* Optional Context:
    "If multiple IDs are linked to the same component, list all related IDs."
    
  Example Input
  Image: [P&ID diagram]
  Text Query: "Identify and list all unique IDs or tag numbers in the diagram. Include the component type if it can be determined."
  Expected Output Example
  ID: P-101
  Component Type: Pump
  ID: V-202
  Component Type: Valve
  ID: T-303
  Component Type: Tank
  ID: P-102
  Component Type: Pump
  ID: C-404
  Component Type: Unknown
"""

# COMMAND ----------

# Process the directory of images
data = [extract_from_image(x, prompt_instruction) for x in directory]

# Convert to Spark DataFrame
if data:
    results_df = spark.createDataFrame(data)
    results_df.show(truncate=False)
else:
    print("No valid data to process.")

# COMMAND ----------

prompt_line_component = """
Line Reference Tags:

"Analyze the provided P&ID diagram and extract all line tags that denote line references. These are typically alphanumeric codes associated with pipelines (e.g., 'L-101' or '12”-LP-001')."
Components Attached to Lines:

"For each identified line tag, list all components (e.g., valves, pumps, instruments) connected to that line. Components include symbols or labels representing equipment or devices."
Association of Tags:

"For every component attached to a line, extract the component’s unique tag or ID (if present). Tags are typically alphanumeric and located near the component symbols."
Output Format:

"Present the findings in a structured format:
Line Tag: [Line Reference Tag]
Attached Component: [Component Type]
Component Tag: [Component Tag or 'No tag found']
Example:
Line Tag: L-101
Attached Component: Valve
Component Tag: V-202
Attached Component: Pump
Component Tag: P-101"
Optional Context:

"If a line tag connects multiple components, list all associated components under the same line tag."
"If a component is attached to multiple lines, repeat the component information under each relevant line tag."
Ignore Unrelated Elements:

"Focus on extracting symbols, IDs, and lines specifically from the diagram. Ignore unrelated text or annotations."
Example Input
Image: [P&ID Diagram]
Text Query: "Extract all line tags, components attached to each line, and their corresponding tags. Provide the information in a structured list."
Expected Output Example
Line Tag: L-101
Attached Component: Valve
Component Tag: V-202
Attached Component: Pump
Component Tag: P-101
Line Tag: L-102
Attached Component: Flow Meter
Component Tag: FM-01
Attached Component: Tank
Component Tag: T-303
Additional Enhancements
Ambiguity Handling:

"If no line tag is found, specify 'No line tag found'."
"If no component tag is found, specify 'No component tag found'."
Confidence Levels:

"Provide a confidence level (high/medium/low) for each extracted line tag or component tag."
Visual Relationships:

"Describe the relationship between components and lines visually, such as 'Valve V-202 is positioned between Pump P-101 and Flow Meter FM-01 on Line L-101'."
"""

# COMMAND ----------

# Process the directory of images
data = [extract_from_image(x, prompt_line_component) for x in directory]

# Convert to Spark DataFrame
if data:
    results_df = spark.createDataFrame(data)
    results_df.show(truncate=False)
else:
    print("No valid data to process.")

# COMMAND ----------

response = client.chat.completions.create(
  model='Qwen-vl-72b',
  messages=[
    {
      "role": "user",
      "content": [
                      {"type": "text", "text": "What can you see in the image?"},
                      {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://images.edrawmax.com/images/knowledge/pid-1-what-is-pid.jpg"
                        },
                      },
                  ],
    }
  ],
  max_tokens=300,
)
print(response.choices[0])

# COMMAND ----------



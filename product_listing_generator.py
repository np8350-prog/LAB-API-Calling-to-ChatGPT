#!/usr/bin/env python
# coding: utf-8

# Important Security Note: Never commit API keys to version control! Always use environment variables or secure storage.
# 
# Checkpoint: Verify that your API key is set correctly and the client initializes without errors.
# 
# 

# In[2]:


import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("Client initialized:", client is not None)


# Step 2: Preparing the Dataset
# Objective: Download and set up the e-commerce product dataset.
# What to do:
# Download the e-commerce product dataset
# Organize product images and metadata
# Create a data structure for products
# 

# In[5]:


# Install: pip install datasets
from datasets import load_dataset
import requests
from PIL import Image
import pandas as pd
from pathlib import Path

# Load dataset from HuggingFace
print("Loading product dataset...")
try:
    # Try loading the dataset
    dataset = load_dataset("ashraq/fashion-product-images-small", split="train[:100]")  # First 100 samples
    print(f"✓ Loaded {len(dataset)} products")

    # Convert to pandas for easier manipulation
    products_df = pd.DataFrame(dataset)
    print(f"Dataset columns: {products_df.columns.tolist()}")

except Exception as e:
    print(f"⚠ Could not load HuggingFace dataset: {e}")
    print("Using local images instead...")

    # Alternative: Use local images
    # Create a products.json file with product information
    products_data = [
        {
            "id": 1,
            "name": "Wireless Headphones",
            "price": 79.99,
            "category": "Electronics",
            "image_path": "images/product1.jpg"
        },
        # Add more products...
    ]

    products_df = pd.DataFrame(products_data)

# Create images directory
images_dir = Path("product_images")
images_dir.mkdir(exist_ok=True)

print(f"\n✓ Dataset prepared!")
print(f"  Total products: {len(products_df)}")


# Step 3: Encoding Images for API
# Objective: Convert product images to base64 format for API transmission.

# In[ ]:


import base64
import io  # needed to convert PIL image object into bytes in memory

def encode_image_to_base64(image_path):
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
    return encoded

# Function above assumes a file path on disk. But dataset has actual PIL Image objects in the "image" column, not paths.
# Add a second version that works directly on a PIL image.
def encode_pil_image_to_base64(pil_image):
    """Encode a PIL Image object to base64 string."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded

# PIL Image object no path.
sample_image = products_df.iloc[0]["image"]
encoded_image = encode_pil_image_to_base64(sample_image)

print(f"Encoded image length: {len(encoded_image)} characters")
print(f"Encoded prefix: {encoded_image[:40]}...")


# WHAT I LEARNED : ERRORs
# encode_image_to_base64(image_path) breaks because I need to check the actual column for image , example had image path but huggingface hold image object ,(always check datasets actual column types)
# 

# Step 4: Creating the Product Listing Prompt
# Objective: Design an effective prompt for generating product listings.
# What to do:
# Create a prompt template
# Include product metadata
# Specify the desired output format

# In[7]:


def create_product_listing_prompt(product_name, price, category, additional_info=None):
    """
    Create a prompt for generating product listings.

    Parameters:
    - product_name: Name of the product
    - price: Price of the product
    - category: Product category
    - additional_info: Optional additional information

    Returns:
    - Formatted prompt string
    """
    prompt = f"""You are an expert e-commerce copywriter. Analyze the product image and create a compelling product listing.

Product Information:
- Name: {product_name}
- Price: ${price:.2f}
- Category: {category}
{f'- Additional Info: {additional_info}' if additional_info else ''}

Please create a professional product listing that includes:

1. **Product Title** (catchy, SEO-friendly, 60 characters max)
2. **Product Description** (detailed, 150-200 words)
   - Highlight key features and benefits
   - Use persuasive language
   - Include relevant details visible in the image
3. **Key Features** (bullet points, 5-7 items)
4. **SEO Keywords** (comma-separated, 10-15 relevant keywords)

Format your response as JSON with the following structure:
{{
    "title": "Product title here",
    "description": "Full description here",
    "features": ["Feature 1", "Feature 2", ...],
    "keywords": "keyword1, keyword2, ..."
}}

Be specific about what you see in the image. Mention colors, materials, design elements, and any distinctive features."""

    return prompt

# Test prompt creation
test_prompt = create_product_listing_prompt(
    product_name="Wireless Bluetooth Headphones",
    price=79.99,
    category="Electronics",
    additional_info="Noise cancelling, 30-hour battery"
)

print("\n" + "="*50)
print("PROMPT TEMPLATE")
print("="*50)
print(test_prompt[:500] + "...")


# Step 5: Calling the ChatGPT API with Vision
# Objective: Send image and text to ChatGPT API and receive response.
# 
# What to do:
# 
# Prepare the API request with image and prompt
# Call the ChatGPT API
# Handle the response
# Parse JSON output
# Expected outcome: You should receive a JSON response with the generated product listing.
# 
# Checkpoint: Verify that:
# 
# API call succeeds
# Response is received
# JSON is parsed correctly
# 

# In[ ]:


def generate_product_listing(product_row):
    """Generate a product listing using ChatGPT vision API."""

    placeholder_price = 29.99

    encoded_image = encode_pil_image_to_base64(product_row["image"])

    prompt = create_product_listing_prompt(
        product_name=product_row["productDisplayName"],
        price=placeholder_price,
        category=product_row["masterCategory"],
        additional_info=f"{product_row['baseColour']} {product_row['articleType']}, {product_row['usage']} wear"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=800
    )

    content = response.choices[0].message.content

    # Got error for the wrapper before parsing , and Json.load() just get raw JSON, Rewrited with cleanups  .

    cleaned_content = content.strip()
    if cleaned_content.startswith("```"):
        cleaned_content = cleaned_content.strip("`")
        cleaned_content = cleaned_content.replace("json\n", "", 1)

    try:
        listing = json.loads(cleaned_content)
    except json.JSONDecodeError:
        listing = {"raw_response": content}

    return listing

sample_listing = generate_product_listing(products_df.iloc[0])
print(json.dumps(sample_listing, indent=2))


# Step 6: Processing Multiple Products
# Objective: Generate listings for multiple products in batch.
# 
# What to do:
# 
# Loop through products
# Generate listing for each
# Save results
# Handle errors gracefully
# Expected outcome: You should process multiple products and save all generated listings.
# 
# Checkpoint: Verify that:
# 
# Multiple products are processed
# Results are saved correctly
# Errors are handled gracefully
# 

# In[11]:


import time

def process_products_batch(products_df, num_products=10):
    """Generate listings for multiple products in batch."""

    results = []

    for idx in range(min(num_products, len(products_df))):
        product_row = products_df.iloc[idx]
        print(f"Processing product {idx + 1}/{num_products}: {product_row['productDisplayName']}")

        try:
            listing = generate_product_listing(product_row)
            listing["product_id"] = int(product_row["id"])
            listing["original_name"] = product_row["productDisplayName"]
            results.append(listing)
            print(f"  ✓ Success")

        except Exception as e:
            print(f"  ⚠ Error: {e}")
            results.append({
                "product_id": int(product_row["id"]),
                "original_name": product_row["productDisplayName"],
                "error": str(e)
            })

        # ADD delay between calls to avoid hitting rate limits good practice for batch API calls.
        time.sleep(1)

    return results

# Run batch for first 10 products
batch_results = process_products_batch(products_df, num_products=10)

print(f"\n✓ Batch complete: {len(batch_results)} products processed")

with open("product_listings.json", "w") as f:
    json.dump(batch_results, f, indent=2)

print(f"✓ Saved {len(batch_results)} listings to product_listings.json")


# Database Setup Guide

This guide explains how to set up your product database with images and SKU information.

## Quick Start

### 1. Prepare Your Product Images

```bash
# Copy your product images to the products folder
mkdir -p database/products
cp /path/to/your/images/*.jpg database/products/
```

### 2. Create products.csv with SKU Information

Create a file at `database/products.csv` with your product information:

```csv
filename,sku_id,product_name,category,price
product_001.jpg,SKU-12345,Blue Cotton Shirt,Clothing,29.99
product_002.jpg,SKU-12346,Red Running Shoes,Footwear,79.99
product_003.jpg,SKU-12347,Wireless Headphones,Electronics,149.99
product_004.jpg,SKU-12348,Coffee Maker,Home Appliances,59.99
```

**CSV Format:**
- `filename`: Name of the image file (must match files in database/products/)
- `sku_id`: Your internal SKU/product ID
- `product_name`: Display name of the product
- `category`: Product category
- `price`: Product price (optional)

### 3. Index Your Database

```bash
./setup-or-start.sh
```

Or manually:

```bash
python index_database.py
```

## Detailed Setup

### Directory Structure

```
mvp/
├── database/
│   ├── products/              # Your product images go here
│   │   ├── product_001.jpg
│   │   ├── product_002.jpg
│   │   └── ...
│   └── products.csv          # SKU mapping file
```

### Image Requirements

**Supported Formats:**
- JPG/JPEG
- PNG
- WEBP
- BMP

**Best Practices:**
- **Resolution**: 800x800 to 2000x2000 pixels (higher is better)
- **File size**: <10MB per image
- **Background**: Clean, preferably white or neutral
- **Lighting**: Consistent, well-lit images
- **Multiple angles**: Add 2-3 images per product from different angles

**Example:**
```
database/products/
  ├── shirt_blue_front.jpg
  ├── shirt_blue_back.jpg
  ├── shirt_blue_side.jpg
  ├── shoes_red_001.jpg
  ├── shoes_red_002.jpg
  └── headphones_black.jpg
```

### Creating products.csv

#### Option 1: Manual Entry

Create `database/products.csv` manually:

```csv
filename,sku_id,product_name,category,price
shirt_blue_front.jpg,SKU-001,Blue Cotton Shirt,Clothing,29.99
shirt_blue_back.jpg,SKU-001,Blue Cotton Shirt,Clothing,29.99
shirt_blue_side.jpg,SKU-001,Blue Cotton Shirt,Clothing,29.99
shoes_red_001.jpg,SKU-002,Red Running Shoes,Footwear,79.99
shoes_red_002.jpg,SKU-002,Red Running Shoes,Footwear,79.99
headphones_black.jpg,SKU-003,Wireless Headphones,Electronics,149.99
```

**Note**: Multiple images can have the same SKU (for different angles).

#### Option 2: Python Script

```python
import pandas as pd
import os

# List all images
images = os.listdir('database/products')
images = [f for f in images if f.endswith(('.jpg', '.jpeg', '.png'))]

# Create DataFrame
data = []
for i, img in enumerate(images, 1):
    data.append({
        'filename': img,
        'sku_id': f'SKU-{i:05d}',
        'product_name': img.replace('.jpg', '').replace('_', ' ').title(),
        'category': 'General',
        'price': '0.00'
    })

df = pd.DataFrame(data)
df.to_csv('database/products.csv', index=False)
print(f"Created products.csv with {len(df)} entries")
```

#### Option 3: From Existing Database

If you have a database, export to CSV:

```python
import pandas as pd
import sqlite3

# Connect to your database
conn = sqlite3.connect('your_database.db')

# Query products
query = """
    SELECT 
        image_filename as filename,
        sku as sku_id,
        name as product_name,
        category,
        price
    FROM products
"""

df = pd.read_sql(query, conn)
df.to_csv('database/products.csv', index=False)
print(f"Exported {len(df)} products")
```

### Large Datasets (1000+ Products)

For large catalogs:

1. **Organize by category:**
```
database/products/
  ├── clothing/
  ├── electronics/
  ├── footwear/
  └── home/
```

2. **Create category-specific CSV files:**
```
database/
  ├── products_clothing.csv
  ├── products_electronics.csv
  └── products_footwear.csv
```

3. **Merge CSV files:**
```python
import pandas as pd
import glob

# Read all CSV files
csv_files = glob.glob('database/products_*.csv')
dfs = [pd.read_csv(f) for f in csv_files]

# Merge
merged = pd.concat(dfs, ignore_index=True)
merged.to_csv('database/products.csv', index=False)
```

### Indexing Options

#### Basic Indexing (Fast)

```bash
python index_database.py
```

Indexes entire images. Best for:
- Single-object images
- Clean backgrounds
- Speed priority

#### Object Detection Mode (Accurate)

```bash
python index_database.py --use-objects
```

Detects and crops objects first. Best for:
- Complex scenes
- Multiple objects
- Cluttered backgrounds
- Higher accuracy needs

### Updating the Database

#### Add New Products

1. Add images to `database/products/`
2. Update `products.csv` with new entries
3. Rebuild index:

```bash
# Via API
curl -X POST http://localhost:5000/api/index/rebuild

# Or via script
python index_database.py
```

#### Remove Products

1. Delete images from `database/products/`
2. Remove entries from `products.csv`
3. Rebuild index:

```bash
python index_database.py
```

### Testing Your Database

After indexing, test with a query:

```bash
# Test with an image
curl -X POST http://localhost:5000/api/search \
  -F "image=@test_image.jpg" \
  | python -m json.tool
```

Expected response:

```json
{
  "success": true,
  "query_time": 0.123,
  "detected_objects": 1,
  "results": [
    {
      "image_path": "database/products/product_001.jpg",
      "similarity": 0.95,
      "sku_id": "SKU-12345",
      "product_name": "Blue Cotton Shirt",
      "category": "Clothing",
      "price": "29.99",
      "rank": 0
    }
  ]
}
```

## Common Issues

### Issue: Images not found during indexing

**Solution**: Ensure images are in `database/products/` directory:
```bash
ls database/products/
```

### Issue: SKU information not appearing in results

**Solutions**:
1. Check CSV exists: `cat database/products.csv`
2. Verify filename matches: Compare CSV filenames with actual files
3. Rebuild index: `python index_database.py`

### Issue: Poor search results

**Solutions**:
1. Use higher quality images
2. Add more images per product (different angles)
3. Enable object detection: `python index_database.py --use-objects`
4. Lower similarity threshold in search

### Issue: Slow indexing

**Solutions**:
1. Reduce image resolution in `config.py`:
   ```python
   MAX_IMAGE_DIMENSION = 1024  # Default: 1920
   ```
2. Use GPU if available:
   ```python
   FAISS_USE_GPU = True
   ```

## Advanced: Batch Image Processing

### Resize Images

```python
from PIL import Image
import os

def resize_images(input_dir, output_dir, max_size=1024):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img = Image.open(os.path.join(input_dir, filename))
            img.thumbnail((max_size, max_size), Image.LANCZOS)
            img.save(os.path.join(output_dir, filename), quality=95)

resize_images('raw_images/', 'database/products/')
```

### Remove Backgrounds

```python
from rembg import remove
from PIL import Image
import os

def remove_backgrounds(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            with open(input_path, 'rb') as f:
                input_img = f.read()
            
            output_img = remove(input_img)
            
            with open(output_path, 'wb') as f:
                f.write(output_img)

# Install: pip install rembg
remove_backgrounds('database/products/', 'database/products_nobg/')
```

## Example Datasets

### E-commerce Fashion

```csv
filename,sku_id,product_name,category,price
dress_red_001.jpg,DRS-001,Red Evening Dress,Women's Fashion,89.99
dress_red_002.jpg,DRS-001,Red Evening Dress,Women's Fashion,89.99
jeans_blue_001.jpg,JNS-002,Classic Blue Jeans,Men's Fashion,49.99
jacket_leather_001.jpg,JKT-003,Leather Jacket,Outerwear,199.99
```

### Electronics

```csv
filename,sku_id,product_name,category,price
laptop_001.jpg,LAP-001,15" Gaming Laptop,Computers,1299.99
phone_001.jpg,PHN-002,Smartphone Pro,Mobile,899.99
headphones_001.jpg,HDP-003,Noise Cancelling Headphones,Audio,299.99
```

### Home & Garden

```csv
filename,sku_id,product_name,category,price
chair_001.jpg,CHR-001,Ergonomic Office Chair,Furniture,249.99
plant_001.jpg,PLT-002,Monstera Plant,Plants,29.99
lamp_001.jpg,LMP-003,LED Desk Lamp,Lighting,39.99
```

## Migration from Old System

If you have an existing database without SKU support:

```python
import os
import pandas as pd

# Get all existing images
images = os.listdir('database/')
images = [f for f in images if f.endswith(('.jpg', '.jpeg', '.png'))]

# Create basic CSV
data = [{
    'filename': img,
    'sku_id': f'SKU-{img}',
    'product_name': img,
    'category': 'Imported',
    'price': ''
} for img in images]

df = pd.DataFrame(data)
df.to_csv('database/products.csv', index=False)

# Move images to products folder
os.makedirs('database/products', exist_ok=True)
for img in images:
    os.rename(f'database/{img}', f'database/products/{img}')

print("Migration complete! Now edit products.csv with real product info.")
```

Then edit `database/products.csv` with actual product information and rebuild the index.


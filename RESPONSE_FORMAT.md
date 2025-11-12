# Response Format - SKU + All Images

## 🎯 New Response Structure

When you search with an image, the system now returns:
- **SKU ID** of the matched product
- **ALL images** for that SKU (not just the matched one)

## 📊 API Response Format

### Request:
```bash
curl -X POST http://localhost:5000/api/search \
  -F "image=@query.jpg"
```

### Response:
```json
{
  "success": true,
  "query_time": 0.123,
  "detected_objects": 1,
  "results": [
    {
      "sku_id": "SKU-001",
      "product_name": "Blue Wireless Headphones",
      "category": "Electronics",
      "price": "99.99",
      "description": "Premium wireless headphones",
      "similarity": 0.95,
      "matched_image": "sample_product_2.jpg",
      "images": [
        "sample_product_1.jpg",
        "sample_product_2.jpg",
        "sample_product_3.jpg"
      ],
      "total_images": 3
    }
  ]
}
```

## 🔑 Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `sku_id` | string | Product SKU identifier |
| `product_name` | string | Product name |
| `category` | string | Product category |
| `price` | string | Product price |
| `description` | string | Product description |
| `similarity` | float | Match score (0-1) |
| `matched_image` | string | Which specific image matched |
| `images` | array | **ALL images for this SKU** |
| `total_images` | int | Count of images |

## 💡 How It Works

### Example Scenario:

**Your Database:**
```json
{
  "products": [
    {
      "sku_id": "SKU-001",
      "product_name": "Headphones",
      "images": ["front.jpg", "back.jpg", "side.jpg"]
    }
  ]
}
```

**You Upload:** `back.jpg` (a photo similar to the back view)

**System Returns:**
- ✅ SKU: `SKU-001`
- ✅ Matched: `back.jpg`
- ✅ **All Images**: `["front.jpg", "back.jpg", "side.jpg"]`

### Benefits:

1. **One image matches → Get whole product**
2. **See all angles** of the matched product
3. **Cleaner response** - One result per SKU (not per image)
4. **Better UX** - Show customer all available views

## 🎨 Web Interface

The web interface now shows:
- **Main matched image** (large)
- **Match percentage**
- **SKU and product info**
- **Gallery of ALL product images** (scrollable)
- **Matched image highlighted** with blue border

## 📝 Python Example

```python
import requests

with open('query.jpg', 'rb') as f:
    r = requests.post('http://localhost:5000/api/search',
                     files={'image': f})

result = r.json()

if result['success'] and result['results']:
    product = result['results'][0]
    
    print(f"Matched Product:")
    print(f"  SKU: {product['sku_id']}")
    print(f"  Name: {product['product_name']}")
    print(f"  Matched Image: {product['matched_image']}")
    print(f"  All Images: {', '.join(product['images'])}")
    print(f"  Total: {product['total_images']} images")
```

Output:
```
Matched Product:
  SKU: SKU-001
  Name: Blue Wireless Headphones
  Matched Image: sample_product_2.jpg
  All Images: sample_product_1.jpg, sample_product_2.jpg, sample_product_3.jpg
  Total: 3 images
```

## 🔄 De-duplication

If multiple images from the same SKU match, you only get ONE result with that SKU (the best match).

**Example:**
- Upload image matches both `front.jpg` (95%) and `back.jpg` (90%)
- You get ONE result: SKU-001 with 95% match
- Response includes ALL images for SKU-001

## ✨ Perfect For:

- E-commerce product search
- Inventory management
- Visual catalog browsing
- Mobile shopping apps

Your customers upload one photo → They see the complete product! 🎉

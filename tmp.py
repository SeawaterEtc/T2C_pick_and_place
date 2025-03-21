import matplotlib.pyplot as plt
import pandas as pd

# Create DataFrame with complete waste categories
data = {
    "Category": ["Recyclable"]*5 + ["Incineration"]*5 + ["Biowaste"]*5,
    "Subcategory": ["Paper", "Plastics", "Metal", "Glass", "Electronics",
                    "Medical Waste", "Hazardous Waste", "Non-recyclable Plastics", 
                    "Textiles", "Rubber", "Food Waste", "Yard Waste",
                    "Animal Waste", "Compostable Paper", "Wood"],
    "Object Examples": [
        "Office Paper, Books, Magazines, Paper Cups, Cardboard, and more",
        "PET Bottles, HDPE Containers, PVC Pipes, Plastic Bags, and more",
        "Steel Cans, Aluminum Cans, Copper Wires, Iron Sheets, and more",
        "Glass Bottles, Jars, Broken Windows, Light Bulbs, and more",
        "Batteries, Circuit Boards, Laptops, Old Phones, and more",
        "Syringes, Expired Medicine, Used Bandages, Gloves, and more",
        "Paint Cans, Pesticides, Chemical Containers, Batteries, and more",
        "Plastic Wraps, Multi-layer Packaging, Styrofoam, and more",
        "Carpet, Old Clothes, Shoes, Upholstery, and more",
        "Rubber Bands, Used Tires, Rubber Gloves, and more",
        "Fruit Peels, Coffee Grounds, Vegetable Scraps, and more",
        "Leaves, Grass Clippings, Dead Plants, and more",
        "Pet Waste, Manure, Bones, and more",
        "Paper Towels, Napkins, Cardboard Food Containers, and more",
        "Sawdust, Wooden Pallets, Tree Stumps, and more"
    ]
}

df = pd.DataFrame(data)

# Create figure
plt.figure(figsize=(16, 10))
ax = plt.gca()
ax.axis('off')

# Create table
table = plt.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc='left',
    loc='center',
    colColours=['#f0f0f0', '#f0f0f0', '#f0f0f0']
)

# Style adjustments
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.5, 1.5)  # Increase cell size

# Set header color
for (i, cell) in enumerate(table.get_celld().values()):
    if i < 3:  # Header cells
        cell.set_facecolor('#404040')
        cell.set_text_props(color='white', weight='bold')

# Save image
output_path = 'T2C_PickAndPlace/Data/waste_categories_table.png'
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Table saved to {output_path}")
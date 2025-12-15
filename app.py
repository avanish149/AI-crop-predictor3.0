import pandas as pd

# load your attached CSV
df = pd.read_csv("crop_recommendation.csv")

# mapping: Crop -> (rates, yield)
crop_data = {
    "Rice":        (25.5, 3850),
    "Maize":       (18.2, 4200),
    "ChickPea":    (65.0, 850),
    "KidneyBeans": (45.0, 2800),
    "PigeonPeas":  (70.0, 720),
    "MothBeans":   (55.0, 450),
    "MungBean":    (60.0, 500),
    "Blackgram":   (58.0, 480),
    "Lentil":      (62.0, 950),
    "Pomegranate": (80.0, 22000),
    "Banana":      (35.0, 35000),
    "Mango":       (45.0, 8500),
    "Grapes":      (120.0, 22000),
    "Watermelon":  (12.0, 25000),
    "Muskmelon":   (15.0, 28000),
    "Apple":       (150.0, 20000),
    "Orange":      (40.0, 15000),
    "Papaya":      (25.0, 35000),
    "Coconut":     (30.0, 14000),
    "Cotton":      (120.0, 800),
    "Jute":        (35.0, 2500),
    "Coffee":      (200.0, 1200),
}

# add columns based on Crop
df[["rates", "yield"]] = df["Crop"].map(crop_data).apply(pd.Series)

# save as the file your code uses
df.to_csv("crop_recommendation.csv", index=False)
print("Updated CSV written to crop_recommendation.csv")

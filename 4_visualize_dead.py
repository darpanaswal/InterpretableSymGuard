import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

feature_df = pd.read_csv("data/feature_activations.csv")
feature_data = feature_df.to_dict(orient="records")

mean_activations = [f["mean_activation"] for f in feature_data]
plt.figure(figsize=(10, 5))
sns.histplot(mean_activations, bins=100, kde=True)
plt.title("Distribution of Mean Feature Activations")
plt.xlabel("Mean Activation")
plt.ylabel("Number of Features")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#import umap
from gensim.models import KeyedVectors
import numpy as np

# Load embeddings
model = KeyedVectors.load("/opt/research/shared/textmining/cspace.kv.bin", 'r') 
model.fill_norms()

# Define main concepts
main_concepts = {
    "chronic_fatigue_syndrome": "Chronic Fatigue Syndrome",
    "idiopathic_pulmonary_fibrosis": "Idiopathic Pulmonary Fibrosis",
    "tuberculosis": "Tuberculosis"
}

# Additional labeled concepts with display names
additional_concepts = {
    "lpqt": "lipoprotein LpqT", # -> tuberculosis
    "gene_1437": "MAPT gene", # -> ipf
    "long_covid": "Long covid" # -> CFS
}


# Get top-10 similar concepts for each main concept
top_similar = {}
all_labels = []
all_vectors = []
colors = []
annotations = []
color_map = {concept: idx for idx, concept in enumerate(main_concepts)}

for concept,label in main_concepts.items():
    if concept not in model:
        print(f"Concept '{concept}' not found in the model.")
        continue

    all_labels.append(label)
    all_vectors.append(model.get_vector(concept, norm=True))
    colors.append(color_map[concept])
    annotations.append(True)  # Annotate only main concepts

    similar = model.most_similar(concept, topn=10)
    for sim_word, _ in similar:
        all_labels.append(sim_word)
        all_vectors.append(model.get_vector(sim_word, norm=True))
        colors.append(color_map[concept])
        annotations.append(False)  # Do not annotate similar concepts


# Add additional labeled concepts
for i, (key, label) in enumerate(additional_concepts.items()):
    if key in model:
        all_labels.append(label)
        all_vectors.append(model.get_vector(key, norm=True))
        colors.append(len(main_concepts)+i)  # Assign different color
        annotations.append(True)
        
# Convert to numpy array
all_vectors = np.array(all_vectors)

# Perform PCA
pca = PCA(n_components=2)
#reducer = umap.UMAP(n_components=2, metric='cosine', spread=2.5, min_dist=0.5)
reduced = pca.fit_transform(all_vectors)
explained_variance = pca.explained_variance_ratio_ * 100

# Plotting
plt.figure(figsize=(6, 4))
scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, cmap='tab10', s=60, alpha=0.8)

# Annotate selected points
for i, label in enumerate(all_labels):
    if annotations[i]:
        plt.annotate(label, (reduced[i, 0], reduced[i, 1]), fontsize=7)


plt.xlabel(f"PC 1 ({explained_variance[0]:.2f}%)")
plt.ylabel(f"PC 1 ({explained_variance[1]:.2f}%)")
plt.grid(True)
plt.tight_layout()
plt.savefig("Figure 2.png", dpi=1200)

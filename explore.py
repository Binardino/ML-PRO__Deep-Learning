from utilities import load_data
import matplotlib.pyplot as plt
import numpy as np

X_train, y_train, X_test, y_test = load_data()

# shapes & types
print("X_train shape:", X_train.shape)   # expected : (n, 64, 64)
print("y_train shape:", y_train.shape)   # expected : (n, 1)
print("X_test  shape:", X_test.shape)
print("y_test  shape:", y_test.shape)
print("dtype pixels  :", X_train.dtype)  # uint8 : values 0-255
print("unique y values:", np.unique(y_train))  # [0 1]

# print 30 images with their labels
plt.figure(figsize=(16, 4))
for i in range(30):
    plt.subplot(2, 15, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f"label: {y_train[i][0]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# ============================================================
# VISUALISING THE RESHAPE (FLATTEN)
# ============================================================
print("\n" + "=" * 60)
print("  UNDERSTANDING RESHAPE (FLATTEN)")
print("=" * 60)

# What does X_train.shape[0] return?
print(f"\nX_train.shape     = {X_train.shape}")
print(f"X_train.shape[0]  = {X_train.shape[0]}  <- number of images (m)")
print(f"X_train.shape[1]  = {X_train.shape[1]}  <- image height in pixels")
print(f"X_train.shape[2]  = {X_train.shape[2]}  <- image width in pixels")

# How reshape(m, -1) computes the flat dimension
total_elements = X_train.shape[0] * X_train.shape[1] * X_train.shape[2]
flat_dim = X_train.shape[1] * X_train.shape[2]
print(f"\nTotal elements    = {X_train.shape[0]} x {X_train.shape[1]} x {X_train.shape[2]} = {total_elements}")
print(f"NumPy computes -1 = {total_elements} / {X_train.shape[0]} = {flat_dim} features per image")

# Perform the reshape
X_train_flat = X_train.reshape(X_train.shape[0], -1)
print(f"\nBEFORE : X_train.shape      = {X_train.shape}       <- 3D (m, height, width)")
print(f"AFTER  : X_train_flat.shape = {X_train_flat.shape}  <- 2D (m, n_features) ✓")

# Visual comparison: original image vs flattened vector
img_index = 0
image_2d = X_train[img_index]        # shape (64, 64) - original image
image_1d = X_train_flat[img_index]   # shape (4096,)  - flattened image

fig, axes = plt.subplots(1, 3, figsize=(16, 4),
                         gridspec_kw={'width_ratios': [1, 0.3, 2]})

# Original 64x64 image
axes[0].imshow(image_2d, cmap='gray')
axes[0].set_title(f"Original image\nshape = {image_2d.shape}", fontsize=12)
axes[0].set_xlabel("64 columns")
axes[0].set_ylabel("64 rows")

# Arrow in the middle
axes[1].axis('off')
axes[1].text(0.5, 0.5, "reshape\n→", fontsize=20, ha='center', va='center',
             fontweight='bold', color='#e74c3c')

# Flattened vector re-displayed as 64x64 to verify data integrity
axes[2].imshow(image_1d.reshape(64, 64), cmap='gray', aspect='auto')
axes[2].set_title(f"Flattened vector reshaped back to 64x64\nflatten shape = ({flat_dim},)", fontsize=12)
axes[2].set_xlabel(f"All {flat_dim} pixels laid end to end, then re-visualised")
axes[2].set_yticks([])

plt.suptitle("Flatten: transforming a 2D image into a 1D vector for the neuron",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Verify that the data is identical before and after
print(f"\nData identical after flatten? {np.array_equal(image_2d.flatten(), image_1d)}")
print("-> flatten does NOT modify the values, it only changes the shape!")